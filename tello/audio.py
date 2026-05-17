"""Audio monitor — pulse-cadence smoke-alarm detector.

Runs a single background thread that opens the default mic via
``sounddevice``, maintains a rolling FFT window, and publishes two
kinds of events on the global event bus:

* ``audio_level``   per-frame DSP snapshot (~5 Hz publish rate). Drives
                    the meter and the "peak freq / pulse count" line in
                    the operator console's Audio card.
* ``audio_alarm``   state transitions ``armed`` -> ``alarm`` -> ``armed``.
                    Phase D wires this to the agent's auto-trigger.

How detection actually works
----------------------------

A residential smoke alarm (UL-217) is a near-pure sinusoid around
3.0-3.2 kHz, played in the **T3 cadence**: 0.5 s on, 0.5 s off, 0.5 s
on, 0.5 s off, 0.5 s on, 1.5 s rest, then repeat. CO alarms (UL-2034)
use the same band, T4 cadence. Cheap phone/speaker recordings are
usually played back as continuous tones at the same frequency.

A first version of this module looked for *sustained* energy in
2.5-4 kHz — which never matched a real T3 alarm (only 0.5 s windows of
tone exist) but did happily trip on TV sibilants, keyboard clicks, and
microwave beeps. This rewrite fixes both ends:

1. Every block, we compute the **spectral peak inside a tight 2.7-3.5
   kHz window** and the *tonality* — peak height vs the in-band mean
   with the peak bin notched out. A pure sinusoid gives ``tonality_db
   >> 20``; broadband sounds give ``tonality_db < 10``. This is what
   discriminates a smoke alarm from speech / static.

2. We segment consecutive tonal frames into **pulses** (closed when
   the tone falls away) and remember the last few pulses' timestamps
   and dominant frequencies.

3. The state machine fires ``alarm`` on either path:
   * **T3-style:** two pulses of 150-900 ms duration within 2.5 s
     whose peak frequencies agree to within ``PULSE_FREQ_TOLERANCE_HZ``.
     This locks onto a real alarm by the second beep — ~1.5 s after
     the first beep starts.
   * **Continuous-tone:** a single strong tone (``tonality_db >=
     STRONG_TONE_DB``) sustained for ``STRONG_TONE_HOLD_SEC``. Catches
     phone recordings and older non-T3 alarms in ~0.6 s.

Threading model
---------------

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

SAMPLE_RATE    = 16_000          # mono 16 kHz; well above the alarm Nyquist
BLOCK_SAMPLES  = 1024            # 64 ms per stream read
WINDOW_SAMPLES = 4096            # 256 ms rolling FFT window
# Tighter than before. UL-217 smoke alarms target 3.0-3.2 kHz; CO alarms
# (UL-2034) sit in the same range. Anything outside this band is almost
# certainly not a residential alarm and would just add false positives.
ALARM_PEAK_BAND_HZ = (2700.0, 3500.0)
# Half-width (in bins) around the detected peak to *exclude* from the
# in-band mean when computing tonality. ~2 bins at 4096-sample FFT @
# 16 kHz = ±8 Hz, which is enough to remove the peak's own skirt without
# eating the rest of the band.
PEAK_NOTCH_BINS = 2

# Publish a level event at most this often (Hz). Block rate is ~16 Hz
# but the UI doesn't need that and ws.send_json on every block is waste.
LEVEL_PUBLISH_HZ = 5.0

# --- per-frame tonal detector --------------------------------------------- #
# A frame is "tonal" when all three of these hold. PEAK_FLOOR_DBFS is the
# absolute floor (rejects silence/noise floor); TONALITY_DB is the
# discriminator that says "this is a tone, not broadband"; the frequency
# range constrains us to the alarm band. Tuned against speech, sibilants,
# keyboard clicks, microwave beeps, and music — none of those produce
# 18 dB of in-band peak prominence in 2.7-3.5 kHz.
PEAK_FLOOR_DBFS   = -55.0
TONALITY_DB       = 18.0

# --- pulse segmentation --------------------------------------------------- #
# A pulse is a stretch of consecutive tonal frames. We accept any pulse
# whose duration is in [MIN, MAX] sec — covers T3's 0.5 s beep plus
# generous slack for the FFT window catching the leading/trailing edges.
PULSE_MIN_SEC = 0.15
PULSE_MAX_SEC = 0.90
# Two pulses with peak frequencies within this tolerance count as the
# "same alarm". Smoke alarms drift well under 100 Hz between beeps.
PULSE_FREQ_TOLERANCE_HZ = 150.0
# How long pulses live in the rolling history. 2.5 s covers T3's worst
# case (a 1.5 s rest between groups, plus an in-progress beep).
PULSE_WINDOW_SEC = 2.5

# --- strong continuous-tone path ------------------------------------------ #
# Catches phone recordings of smoke alarms that don't bother with T3
# cadence — they just play the 3 kHz tone continuously. We require a
# higher tonality bar than the per-frame floor and a 600 ms hold so
# random musical sustains don't trip us.
STRONG_TONE_DB       = 25.0
STRONG_TONE_HOLD_SEC = 0.60

# --- state-machine hysteresis --------------------------------------------- #
# How long the alarm has to be gone before we drop back to "armed". T3
# has a 1.5 s rest between groups, so 2.5 s gives margin.
ALARM_CLEAR_SEC = 2.5


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class AudioStatus:
    """Snapshot returned by ``/api/audio/status`` and bus events."""

    enabled: bool
    state: str  # "idle" | "armed" | "alarm" | "error"
    tonality_db: float | None
    peak_freq_hz: float | None
    broadband_db: float | None
    pulses_recent: int
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
        # Last DSP snapshot — what the meter and the peak-freq line show.
        self._last_tonality: float | None = None
        self._last_peak_hz: float | None = None
        self._last_broadband: float | None = None
        self._pulses_recent: int = 0
        self._device_name: str | None = None
        self._last_error: str | None = None
        # Manual simulate: when set in the future, treat as alarm until ts.
        self._simulate_until: float = 0.0

    # --- public API ---------------------------------------------------- #

    def status(self) -> AudioStatus:
        return AudioStatus(
            enabled=self._enabled,
            state=self._state,
            tonality_db=self._last_tonality,
            peak_freq_hz=self._last_peak_hz,
            broadband_db=self._last_broadband,
            pulses_recent=self._pulses_recent,
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

    def simulate_alarm(
        self,
        duration_sec: float = 4.0,
        publish: bool = True,
    ) -> dict[str, Any]:
        """Manually fire an alarm event for ``duration_sec``.

        Lets you exercise the agent's auto-trigger without a real smoke
        alarm on hand. The downstream path (agent activation, mission
        flow) is identical to a detected alarm.
        """
        until = time.time() + max(0.5, duration_sec)
        self._simulate_until = until
        if publish:
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
        window.extend(np.zeros(WINDOW_SAMPLES, dtype=np.float32))

        # Precompute FFT bin -> Hz mapping and the alarm-peak-band mask.
        freqs = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE)
        band_mask = (freqs >= ALARM_PEAK_BAND_HZ[0]) & (freqs <= ALARM_PEAK_BAND_HZ[1])
        band_idx = np.flatnonzero(band_mask)
        hann = np.hanning(WINDOW_SAMPLES).astype(np.float32)

        # Pulse history: (close_ts, dominant_freq_hz) per closed pulse.
        pulses: deque[tuple[float, float]] = deque()
        in_pulse = False
        pulse_start = 0.0
        pulse_freqs: list[float] = []

        strong_tone_start: float | None = None
        clear_since: float | None = None
        last_publish = 0.0

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
            spectrum = np.abs(np.fft.rfft(arr * hann))

            # --- DSP: locate the dominant tone inside the alarm peak band #
            band_spectrum = spectrum[band_mask]
            peak_local = int(np.argmax(band_spectrum))
            peak_idx_global = int(band_idx[peak_local])
            peak_freq = float(freqs[peak_idx_global])
            peak_mag = float(band_spectrum[peak_local])

            # Notch ±N bins around the peak when computing the in-band
            # mean — gives us "peak vs everything else in the band", which
            # is what really separates a tone from broadband content.
            mean_excl_peak = _band_mean_excluding_peak(
                band_spectrum, peak_local, PEAK_NOTCH_BINS
            )
            peak_db     = 20.0 * math.log10(peak_mag       + 1e-12)
            mean_db     = 20.0 * math.log10(mean_excl_peak + 1e-12)
            tonality_db = peak_db - mean_db
            broad_db    = 20.0 * math.log10(
                math.sqrt(float(np.mean(spectrum ** 2))) + 1e-12
            )

            self._last_tonality  = tonality_db
            self._last_peak_hz   = peak_freq
            self._last_broadband = broad_db

            now = time.time()

            # --- per-frame "is this a tone?" decision ------------------- #
            is_tone = (
                peak_db >= PEAK_FLOOR_DBFS
                and tonality_db >= TONALITY_DB
                and ALARM_PEAK_BAND_HZ[0] <= peak_freq <= ALARM_PEAK_BAND_HZ[1]
            )

            # --- pulse segmentation ------------------------------------- #
            if is_tone:
                if not in_pulse:
                    in_pulse = True
                    pulse_start = now
                    pulse_freqs = []
                pulse_freqs.append(peak_freq)
            else:
                if in_pulse:
                    duration = now - pulse_start
                    if PULSE_MIN_SEC <= duration <= PULSE_MAX_SEC and pulse_freqs:
                        pulses.append((now, float(np.mean(pulse_freqs))))
                    in_pulse = False
                    pulse_freqs = []

            # Trim pulses outside the rolling window.
            cutoff = now - PULSE_WINDOW_SEC
            while pulses and pulses[0][0] < cutoff:
                pulses.popleft()
            self._pulses_recent = len(pulses)

            # --- continuous strong-tone tracking ------------------------ #
            if is_tone and tonality_db >= STRONG_TONE_DB:
                if strong_tone_start is None:
                    strong_tone_start = now
            else:
                strong_tone_start = None

            # --- alarm decision: cadence OR continuous OR simulate ------ #
            simulating = now < self._simulate_until
            trigger = False
            reason = ""
            if simulating:
                trigger = True
                reason = "simulate button"
            elif len(pulses) >= 2 and _freqs_agree(pulses):
                trigger = True
                avg = sum(f for _, f in pulses) / len(pulses)
                reason = (
                    f"{len(pulses)} matching pulses around {avg:.0f} Hz "
                    f"(tonality {tonality_db:+.0f} dB)"
                )
            elif strong_tone_start is not None and (now - strong_tone_start) >= STRONG_TONE_HOLD_SEC:
                trigger = True
                reason = (
                    f"sustained {peak_freq:.0f} Hz tone "
                    f"(tonality {tonality_db:+.0f} dB)"
                )

            # --- state machine ------------------------------------------ #
            if trigger:
                clear_since = None
                if self._state != "alarm":
                    self._state = "alarm"
                    bus.publish_threadsafe(
                        {
                            "type": "audio_alarm",
                            "state": "alarm",
                            "source": "manual" if simulating else "detector",
                            "tonality_db":  tonality_db,
                            "peak_freq_hz": peak_freq,
                            "broadband_db": broad_db,
                            "pulses_recent": len(pulses),
                            "reason": reason,
                        }
                    )
            else:
                if self._state == "alarm":
                    if clear_since is None:
                        clear_since = now
                    elif (now - clear_since) >= ALARM_CLEAR_SEC:
                        self._state = "armed"
                        bus.publish_threadsafe(
                            {
                                "type": "audio_alarm",
                                "state": "armed",
                                "source": "detector",
                                "reason": (
                                    f"tone gone for >{ALARM_CLEAR_SEC:.1f} s"
                                ),
                            }
                        )
                        clear_since = None

            if now - last_publish >= 1.0 / LEVEL_PUBLISH_HZ:
                last_publish = now
                bus.publish_threadsafe(
                    {
                        "type": "audio_level",
                        "tonality_db":  tonality_db,
                        "peak_freq_hz": peak_freq,
                        "broadband_db": broad_db,
                        "pulses_recent": len(pulses),
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


# --------------------------------------------------------------------------- #
# Pure functions — easy to unit-test in isolation
# --------------------------------------------------------------------------- #


def _band_mean_excluding_peak(
    band_spectrum: np.ndarray, peak_local_idx: int, notch_bins: int
) -> float:
    """Mean magnitude in the band with ±``notch_bins`` around the peak masked out.

    Falls back to the full mean if notching would leave fewer than 3 bins
    — protects us when the peak band itself is very narrow.
    """
    n = band_spectrum.size
    lo = max(0, peak_local_idx - notch_bins)
    hi = min(n, peak_local_idx + notch_bins + 1)
    keep = np.ones(n, dtype=bool)
    keep[lo:hi] = False
    if keep.sum() < 3:
        return float(np.mean(band_spectrum) + 1e-12)
    return float(np.mean(band_spectrum[keep]) + 1e-12)


def _freqs_agree(pulses: deque[tuple[float, float]]) -> bool:
    """True if the most recent pulses' peak frequencies are within tolerance.

    We compare the last min(3, len) pulses so a stray off-frequency pulse
    from earlier in the window doesn't spoil an otherwise clean cadence.
    """
    recent = [f for _, f in list(pulses)[-3:]]
    if len(recent) < 2:
        return False
    return (max(recent) - min(recent)) <= PULSE_FREQ_TOLERANCE_HZ


# Process-wide singleton (single dashboard, single mic).
monitor = AudioMonitor()
