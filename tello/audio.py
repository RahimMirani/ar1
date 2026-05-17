"""Audio monitor — FFT smoke-alarm detector.

Runs a single background thread that opens the default mic via
``sounddevice``, maintains a rolling FFT window, and publishes two kinds
of events on the global event bus:

* ``audio_level``   alarm-band dB + broadband dB, ~5 Hz, drives the meter
                    in the operator console.
* ``audio_alarm``   state transitions ``armed`` -> ``alarm`` -> ``armed``.
                    Phase D wires this to the agent's auto-trigger.

We do **not** attempt to match the UL-217 T3 cadence (0.5 s on, 0.5 s off
x3, 1.5 s rest). A sustained 2.5-4 kHz tone for >= ALARM_HOLD_SEC is a
fine hackathon heuristic, and the dashboard's "Simulate alarm" button
bypasses the detector entirely — same event, same downstream agent path.

Threading model:

* ``AudioMonitor.start()`` spawns ``tello-audio`` and returns.
* The thread owns the ``sd.InputStream`` and blocks on ``stream.read``.
* All bus publishes go through ``bus.publish_threadsafe`` so the loop
  thread does the actual ``ws.send_json``.
* ``stop()`` flips the enabled flag, closes the stream and joins.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except OSError as exc:  # PortAudio not available — surface a clear error.
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR: Exception | None = exc
else:
    _SD_IMPORT_ERROR = None

from events import bus

logger = logging.getLogger("tello.audio")

# --------------------------------------------------------------------------- #
# DSP knobs
# --------------------------------------------------------------------------- #

SAMPLE_RATE   = 16_000           # mono 16 kHz is plenty for a 3 kHz tone
BLOCK_SAMPLES = 1024             # 64 ms per stream read
WINDOW_SAMPLES = 4096            # 256 ms rolling FFT window
# Smoke alarms (UL-217) sit on a ~3.1 kHz tone. We open the band a bit
# wider so other alarm patterns (CO, fire panel) also trip the detector.
ALARM_BAND_HZ = (2500.0, 4000.0)

# Publish a level event at most this often (Hz). Block rate is ~16 Hz but
# the UI doesn't need that and ws.send_json on every block is wasteful.
LEVEL_PUBLISH_HZ = 5.0

# Detection thresholds. The first is an absolute floor (so a quiet room
# doesn't trip on noise), the second is the ratio between the 3 kHz band
# and the broadband RMS — a tonal alarm is much louder in its band than
# the rest of the spectrum, so this ratio is the discriminator.
ALARM_BAND_DBFS_MIN = -45.0      # band loudness floor
ALARM_RATIO_DB_MIN  = 12.0       # band - broadband in dB

ALARM_HOLD_SEC  = 1.0            # must be hot this long to commit to "alarm"
ALARM_CLEAR_SEC = 2.0            # must be quiet this long to drop back


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class AudioStatus:
    """Snapshot returned by ``/api/audio/status`` and bus events."""

    enabled: bool
    state: str  # "idle" | "armed" | "alarm" | "error"
    alarm_band_db: float | None
    broadband_db: float | None
    device: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #


class AudioMonitor:
    def __init__(self) -> None:
        self._enabled = False
        self._state = "idle"  # public state
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_level: tuple[float, float] | None = None
        self._device_name: str | None = None
        self._last_error: str | None = None
        # Manual simulate: when set in the future, treat as alarm until ts.
        self._simulate_until: float = 0.0

    # --- public API ---------------------------------------------------- #

    def status(self) -> AudioStatus:
        band, broad = self._last_level if self._last_level else (None, None)
        return AudioStatus(
            enabled=self._enabled,
            state=self._state,
            alarm_band_db=band,
            broadband_db=broad,
            device=self._device_name,
            error=self._last_error,
        )

    def start(self) -> AudioStatus:
        if self._enabled:
            return self.status()
        if sd is None:
            self._last_error = (
                f"sounddevice import failed: {_SD_IMPORT_ERROR}. "
                "On Windows the bundled PortAudio DLL ships with the wheel; "
                "make sure 'uv sync' finished without errors."
            )
            self._state = "error"
            return self.status()
        self._stop_event.clear()
        self._enabled = True
        self._state = "armed"
        self._last_error = None
        t = threading.Thread(target=self._run, name="tello-audio", daemon=True)
        self._thread = t
        t.start()
        bus.publish_threadsafe(
            {"type": "audio_alarm", "state": "armed", "source": "monitor"}
        )
        return self.status()

    def stop(self) -> AudioStatus:
        if not self._enabled:
            return self.status()
        self._enabled = False
        self._stop_event.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=2.0)
        self._state = "idle"
        bus.publish_threadsafe(
            {"type": "audio_alarm", "state": "idle", "source": "monitor"}
        )
        return self.status()

    def simulate_alarm(self, duration_sec: float = 4.0) -> dict[str, Any]:
        """Manually fire an alarm event for ``duration_sec``.

        Lets you exercise the agent's auto-trigger without a real smoke
        alarm on hand. The downstream path (agent activation, mission
        flow) is identical to a detected alarm.
        """
        until = time.time() + max(0.5, duration_sec)
        self._simulate_until = until
        bus.publish_threadsafe(
            {
                "type": "audio_alarm",
                "state": "alarm",
                "source": "manual",
                "until": until,
                "reason": "operator pressed Simulate alarm",
            }
        )
        return {"ok": True, "until": until, "duration_sec": duration_sec}

    # --- worker -------------------------------------------------------- #

    def _run(self) -> None:
        try:
            device_idx = os.getenv("FIREDRONE_AUDIO_DEVICE") or None
            if device_idx is not None and device_idx.strip().isdigit():
                device_idx = int(device_idx)
            logger.info("opening mic (device=%s)", device_idx if device_idx is not None else "default")
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SAMPLES,
                dtype="float32",
                device=device_idx,
            ) as stream:
                self._device_name = self._describe_device(stream.device)
                self._loop(stream)
        except Exception as exc:
            self._last_error = f"audio stream failed: {exc}"
            self._state = "error"
            logger.warning(self._last_error)
            bus.publish_threadsafe(
                {"type": "audio_alarm", "state": "error", "error": self._last_error}
            )

    def _loop(self, stream: "sd.InputStream") -> None:
        window = deque(maxlen=WINDOW_SAMPLES)
        # Pre-fill with silence so the first FFT is well-defined.
        window.extend(np.zeros(WINDOW_SAMPLES, dtype=np.float32))

        # FFT bin -> Hz lookup; precompute the alarm-band slice.
        freqs = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE)
        band_mask = (freqs >= ALARM_BAND_HZ[0]) & (freqs <= ALARM_BAND_HZ[1])

        last_publish = 0.0
        hot_since: float | None = None
        cold_since: float | None = None

        while not self._stop_event.is_set():
            try:
                data, _overflow = stream.read(BLOCK_SAMPLES)
            except Exception as exc:
                self._last_error = f"stream.read failed: {exc}"
                logger.warning(self._last_error)
                break
            samples = data[:, 0]
            window.extend(samples)
            arr = np.asarray(window, dtype=np.float32)
            spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
            band_energy  = float(np.sqrt(np.mean(spectrum[band_mask] ** 2)) + 1e-12)
            broad_energy = float(np.sqrt(np.mean(spectrum ** 2)) + 1e-12)
            band_db  = 20.0 * math.log10(band_energy)
            broad_db = 20.0 * math.log10(broad_energy)
            self._last_level = (band_db, broad_db)

            now = time.time()

            # Heuristic detection.
            is_hot = (
                band_db >= ALARM_BAND_DBFS_MIN
                and (band_db - broad_db) >= ALARM_RATIO_DB_MIN
            )
            simulating = now < self._simulate_until
            if simulating:
                is_hot = True

            if is_hot:
                cold_since = None
                if hot_since is None:
                    hot_since = now
                if (
                    self._state != "alarm"
                    and (now - hot_since) >= ALARM_HOLD_SEC
                ):
                    self._state = "alarm"
                    bus.publish_threadsafe(
                        {
                            "type": "audio_alarm",
                            "state": "alarm",
                            "source": "manual" if simulating else "detector",
                            "alarm_band_db": band_db,
                            "broadband_db": broad_db,
                            "reason": (
                                "simulate button"
                                if simulating
                                else f"sustained 2.5-4 kHz tone "
                                f"({band_db:+.0f} dB, ratio {band_db - broad_db:+.0f} dB)"
                            ),
                        }
                    )
            else:
                hot_since = None
                if cold_since is None:
                    cold_since = now
                if (
                    self._state == "alarm"
                    and (now - cold_since) >= ALARM_CLEAR_SEC
                ):
                    self._state = "armed"
                    bus.publish_threadsafe(
                        {
                            "type": "audio_alarm",
                            "state": "armed",
                            "source": "detector",
                            "reason": "tone gone for >2 s",
                        }
                    )

            if now - last_publish >= 1.0 / LEVEL_PUBLISH_HZ:
                last_publish = now
                bus.publish_threadsafe(
                    {
                        "type": "audio_level",
                        "alarm_band_db": band_db,
                        "broadband_db": broad_db,
                        "state": self._state,
                    }
                )

    @staticmethod
    def _describe_device(idx) -> str | None:
        if sd is None or idx is None:
            return None
        try:
            info = sd.query_devices(idx)
            return f"{info['name']} ({idx})"
        except Exception:
            return str(idx)


# Process-wide singleton (single dashboard, single mic).
monitor = AudioMonitor()
