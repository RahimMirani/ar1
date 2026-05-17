"""High-level wrapper around `djitellopy.Tello`.

The only module in the `tello/` project allowed to make raw SDK calls. The
FastAPI server, smoke scripts, and (later) the agent code all go through the
:class:`Drone` class so connection state, telemetry caching, and unit
conversion live in one place.

Two control modes coexist:

* **RC velocity** (used by the dashboard): :meth:`set_velocity` sets a
  ``(lr, fb, ud, yaw)`` vector in [-100, 100] units. A background thread
  forwards it to the drone at ~20 Hz via the non-blocking ``rc`` SDK
  command, so the drone reacts within tens of milliseconds. Holding
  multiple keys composes naturally (e.g. forward + right = diagonal).
* **Discrete distance** (used by the future agent): :meth:`move` and
  :meth:`rotate` send blocking distance/yaw commands. The drone reports
  ``ok`` only after physically completing the move. Convenient for
  scripted waypoints but unsuitable for live teleop.

Conventions
-----------
* Translation velocity components ``lr``, ``fb``, ``ud`` and ``yaw`` are
  integers in **[-100, 100]** (Tello SDK ``rc`` percentages).
* Movement distances are in **centimeters**, valid range 20-500 cm.
* Yaw distances are in **degrees**, 1-360.
* Flips take a single character: ``"f"``, ``"b"``, ``"l"``, ``"r"``.
* No flight-time cap and no low-battery auto-land in milestone 1 — only the
  WebSocket-disconnect emergency stop in `main.py` provides automatic safety.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import av as _av
import cv2
import numpy as np
from djitellopy import Tello
from djitellopy.tello import BackgroundFrameRead, TelloException

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level link-health counters
#
# Counters are incremented from threads owned by djitellopy (state UDP
# receiver, video decoder) so they live at module scope behind a single lock.
# The `Drone` instance samples them periodically to compute packet loss and
# video-decode error rates. There is one Tello per process in this project so
# globals are fine; if that ever changes we'd key by drone address.
# --------------------------------------------------------------------------- #

_link_counters_lock = threading.Lock()
_state_packets_total: int = 0
_last_state_packet_mono: float = 0.0  # monotonic ts of last state packet
_video_errors_total: int = 0


_state_counter_patched = False


def _patch_state_packet_counter() -> None:
    """Monkey-patch ``Tello.parse_state`` to count incoming state packets.

    The Tello broadcasts state at ~10 Hz on UDP 8890; djitellopy's static
    ``udp_state_receiver`` thread calls ``parse_state`` once per packet. By
    wrapping ``parse_state`` we get a free packet counter without touching
    the receiver thread, and the rate it reports is a direct measure of how
    many state datagrams are surviving the WiFi link.
    """

    global _state_counter_patched
    if _state_counter_patched:
        return

    orig = Tello.parse_state

    def parse_state(state: str):
        result = orig(state)
        # ``parse_state`` returns ``{}`` for the literal "ok" handshake reply
        # that some firmware versions echo onto the state channel. Only count
        # real telemetry packets (non-empty dicts).
        if result:
            global _state_packets_total, _last_state_packet_mono
            with _link_counters_lock:
                _state_packets_total += 1
                _last_state_packet_mono = time.monotonic()
        return result

    Tello.parse_state = staticmethod(parse_state)
    _state_counter_patched = True


_patch_state_packet_counter()


# --------------------------------------------------------------------------- #
# Low-latency video decoder
#
# djitellopy opens its H.264 UDP stream via PyAV's ``av.open`` and does not
# expose any way to pass decoder options. Out of the box libav buffers several
# frames for B-frame reordering and "smooth playback", which adds 100-300 ms
# of latency to the dashboard video. We monkey-patch ``av.open`` exactly once
# at import time to inject low-delay options whenever the URL is a UDP stream.
# Non-UDP callers are unaffected.
#
# A *small* ``probesize`` breaks H.264 on UDP: the demuxer never sees enough of
# the stream to pick up SPS/PPS, and ``av.codec`` raises ``InvalidDataError``
# on the first packets. Keep probe/analyze large enough for extradata, while
# still using ``nobuffer`` / ``low_delay`` for live view latency.
# --------------------------------------------------------------------------- #

_LOW_LATENCY_OPTIONS = {
    "fflags": "nobuffer+discardcorrupt",
    "flags": "low_delay",
    "probesize": "5000000",
    "analyzeduration": "1000000",
    "max_delay": "0",
}

_orig_av_open = _av.open


def _patched_av_open(file=None, *args, **kwargs):
    target = file if file is not None else (args[0] if args else None)
    if isinstance(target, str) and target.startswith("udp://"):
        opts = dict(_LOW_LATENCY_OPTIONS)
        if kwargs.get("options"):
            opts.update(kwargs["options"])
        kwargs["options"] = opts
    return _orig_av_open(file, *args, **kwargs)


if getattr(_av.open, "__name__", "") != "_patched_av_open":
    _av.open = _patched_av_open


def _patch_resilient_video_decode() -> None:
    """djitellopy's frame loop uses ``container.decode()``; a single bad H.264
    packet kills the worker thread. Demux packet-by-packet and skip corrupt
    NAL units so the feed recovers. Each skipped (corrupt) frame is counted
    in ``_video_errors_total`` — the rate of these is a sensitive proxy for
    a degrading WiFi link, often visible before the command channel breaks.
    """

    def update_frame(self) -> None:
        try:
            for packet in self.container.demux(video=0):
                if self.stopped:
                    self.container.close()
                    break
                try:
                    for frame in packet.decode():
                        if self.with_queue:
                            self.frames.append(np.array(frame.to_image()))
                        else:
                            self.frame = np.array(frame.to_image())
                except _av.error.InvalidDataError:
                    global _video_errors_total
                    with _link_counters_lock:
                        _video_errors_total += 1
                    continue
        except _av.error.ExitError:
            raise TelloException(
                "Do not have enough frames for decoding, please try again "
                "or increase video fps before get_frame_read()"
            )

    BackgroundFrameRead.update_frame = update_frame  # type: ignore[assignment]


_patch_resilient_video_decode()


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Per-command movement bounds in cm. The Tello SDK rejects values outside this.
MIN_MOVE_CM = 20
MAX_MOVE_CM = 500

# Per-command yaw bounds in degrees.
MIN_YAW_DEG = 1
MAX_YAW_DEG = 360

# Step sizes used by the (now unused) discrete dashboard commands.
DEFAULT_STEP_CM = 30
DEFAULT_YAW_DEG = 30

# RC velocity bounds (Tello SDK ``rc`` percentages).
MIN_VELOCITY = -100
MAX_VELOCITY = 100

# Rate at which the RC background thread forwards the velocity vector. The
# Tello firmware auto-hovers if it doesn't receive an ``rc`` command for ~500
# ms, so we have to send continuously while flying. 20 Hz is comfortably above
# that threshold and well below djitellopy's internal rate limit.
RC_LOOP_HZ = 20.0

VALID_FLIP_DIRECTIONS = {"f", "b", "l", "r"}
VALID_MOVE_DIRECTIONS = {"forward", "back", "left", "right", "up", "down"}


@dataclass
class DroneSnapshot:
    """Immutable read of latest cached state + telemetry, JSON-serializable."""

    connected: bool
    streaming: bool
    flying: bool
    last_status: str
    last_error: Optional[str]
    velocity: dict[str, int] = field(default_factory=lambda: {"lr": 0, "fb": 0, "ud": 0, "yaw": 0})
    telemetry: dict[str, Any] = field(default_factory=dict)


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


class Drone:
    """Semantic wrapper around `djitellopy.Tello`.

    Usage::

        with Drone() as drone:
            drone.connect()
            drone.start_stream()
            drone.takeoff()
            drone.set_velocity(0, 60, 0, 0)   # fly forward at 60% stick
            time.sleep(2.0)
            drone.set_velocity(0, 0, 0, 0)    # stop / hover
            drone.land()
    """

    def __init__(self, host: str = "192.168.10.1") -> None:
        # djitellopy is loud by default; cut it back to warnings.
        Tello.LOGGER.setLevel(logging.WARNING)

        self._tello = Tello(host=host)
        self._lock = threading.RLock()
        self._closed = False

        self._connected = False
        self._streaming = False
        self._flying = False
        self._last_status = "DISCONNECTED"
        self._last_error: Optional[str] = None
        self._last_telemetry: dict[str, Any] = {}

        # RC velocity vector: (left_right, forward_back, up_down, yaw).
        self._velocity: tuple[int, int, int, int] = (0, 0, 0, 0)

        self._telemetry_thread: Optional[threading.Thread] = None
        self._rc_thread: Optional[threading.Thread] = None
        self._wifi_thread: Optional[threading.Thread] = None

        # Link-health state — populated by the wifi/telemetry loops, sampled
        # by ``_compute_link()`` for the dashboard.
        self._link_lock = threading.Lock()
        self._link_snr_db: Optional[int] = None      # last successful wifi? value
        self._link_snr_ts: float = 0.0               # monotonic ts of the read
        self._link_rtt_ms: Optional[float] = None    # round-trip time of wifi?
        # Sliding-window history of (mono_ts, state_packets_total,
        # video_errors_total). Sampled at 2 Hz; ~5 sec window is enough to
        # smooth out the noise without lagging operator perception.
        self._link_history: deque[tuple[float, int, int]] = deque(maxlen=10)

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "Drone":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Best-effort cleanup: stop stream, land if flying, end connection."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._velocity = (0, 0, 0, 0)
            if self._flying:
                try:
                    self._tello.land()
                except Exception:
                    try:
                        self._tello.emergency()
                    except Exception:
                        pass
            if self._streaming:
                try:
                    self._tello.streamoff()
                except Exception:
                    pass
            try:
                self._tello.end()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Connection / stream
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        with self._lock:
            self._tello.connect()
            self._connected = True
            self._last_status = "CONNECTED"
            self._last_error = None
        self._start_background_threads()

    def start_stream(self) -> None:
        with self._lock:
            if not self._connected:
                raise RuntimeError("Drone not connected; call connect() first.")
            if self._streaming:
                return
            self._tello.streamon()
            self._streaming = True
            self._last_status = "STREAMING"

    def stop_stream(self) -> None:
        with self._lock:
            if not self._streaming:
                return
            try:
                self._tello.streamoff()
            finally:
                self._streaming = False

    def get_frame(self):
        """Return the latest decoded frame as a **BGR** numpy array, or None.

        djitellopy decodes Tello's H.264 stream into RGB (via ``PIL.Image``),
        but every downstream consumer in this project uses OpenCV convention
        (BGR) — ``cv2.imencode`` for the MJPEG stream, ``cv2.imshow`` for the
        smoke viewer, and OpenCV in general for any future vision code. We
        do the single conversion here so callers never see the mismatch.
        """
        if not self._streaming:
            return None
        try:
            rgb = self._tello.get_frame_read().frame
            if rgb is None:
                return None
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("get_frame failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Flight commands (one-shot)
    # ------------------------------------------------------------------ #

    def takeoff(self) -> None:
        with self._lock:
            self._tello.takeoff()
            self._flying = True
            self._last_status = "TAKEOFF"

    def land(self) -> None:
        with self._lock:
            self._velocity = (0, 0, 0, 0)
            self._tello.land()
            self._flying = False
            self._last_status = "LANDED"

    def emergency(self) -> None:
        """Cut motors immediately. Never raises."""
        try:
            self._tello.emergency()
        except Exception as exc:
            logger.warning("emergency() raised: %s", exc)
        with self._lock:
            self._velocity = (0, 0, 0, 0)
            self._flying = False
            self._last_status = "EMERGENCY"

    def flip(self, direction: str) -> None:
        """Flip in one of the four cardinal directions. Needs battery > 50%."""
        if direction not in VALID_FLIP_DIRECTIONS:
            raise ValueError(f"Invalid flip direction: {direction!r}")
        with self._lock:
            self._tello.flip(direction)
            self._last_status = f"FLIP {direction.upper()}"

    # ------------------------------------------------------------------ #
    # RC velocity control (live teleop)
    # ------------------------------------------------------------------ #

    def set_velocity(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Update the live velocity vector. Non-blocking; returns immediately.

        Each component is clamped to ``[-100, 100]``. The background RC thread
        forwards the vector to the drone at ~20 Hz, so the drone responds in
        tens of milliseconds. Send ``(0, 0, 0, 0)`` to stop and hover.
        """
        lr  = _clamp_int(lr,  MIN_VELOCITY, MAX_VELOCITY)
        fb  = _clamp_int(fb,  MIN_VELOCITY, MAX_VELOCITY)
        ud  = _clamp_int(ud,  MIN_VELOCITY, MAX_VELOCITY)
        yaw = _clamp_int(yaw, MIN_VELOCITY, MAX_VELOCITY)
        with self._lock:
            self._velocity = (lr, fb, ud, yaw)
            self._last_status = f"VEL lr={lr} fb={fb} ud={ud} yaw={yaw}"
        # Fire one packet immediately so the drone reacts without waiting for
        # the next 20 Hz tick. djitellopy rate-limits internally, so this is
        # safe to call frequently.
        if self._flying and not self._closed:
            try:
                self._tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as exc:
                logger.debug("send_rc_control (immediate) failed: %s", exc)

    def stop_velocity(self) -> None:
        """Convenience: zero out the velocity vector."""
        self.set_velocity(0, 0, 0, 0)

    # ------------------------------------------------------------------ #
    # Discrete distance commands (for the future agent, not live teleop)
    # ------------------------------------------------------------------ #

    def move(self, direction: str, distance_cm: int = DEFAULT_STEP_CM) -> None:
        """Blocking: move ``direction`` by ``distance_cm`` (20-500 cm)."""
        if direction not in VALID_MOVE_DIRECTIONS:
            raise ValueError(f"Invalid move direction: {direction!r}")
        distance_cm = _clamp_int(distance_cm, MIN_MOVE_CM, MAX_MOVE_CM)
        method = getattr(self._tello, f"move_{direction}")
        with self._lock:
            method(distance_cm)
            self._last_status = f"MOVE {direction.upper()} {distance_cm}cm"

    def rotate(self, direction: str, degrees: int = DEFAULT_YAW_DEG) -> None:
        """Blocking: yaw ``cw``/``ccw`` by ``degrees`` (1-360)."""
        degrees = _clamp_int(degrees, MIN_YAW_DEG, MAX_YAW_DEG)
        if direction == "cw":
            method = self._tello.rotate_clockwise
        elif direction == "ccw":
            method = self._tello.rotate_counter_clockwise
        else:
            raise ValueError(f"Invalid rotation direction: {direction!r}")
        with self._lock:
            method(degrees)
            self._last_status = f"ROTATE {direction.upper()} {degrees}deg"

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def snapshot(self) -> DroneSnapshot:
        with self._lock:
            lr, fb, ud, yaw = self._velocity
            return DroneSnapshot(
                connected=self._connected,
                streaming=self._streaming,
                flying=self._flying,
                last_status=self._last_status,
                last_error=self._last_error,
                velocity={"lr": lr, "fb": fb, "ud": ud, "yaw": yaw},
                telemetry=dict(self._last_telemetry),
            )

    def set_error(self, message: str) -> None:
        """Record a user-visible error string for the dashboard."""
        with self._lock:
            self._last_error = message

    def clear_error(self) -> None:
        with self._lock:
            self._last_error = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _start_background_threads(self) -> None:
        if self._telemetry_thread is None or not self._telemetry_thread.is_alive():
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name="tello-telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

        if self._rc_thread is None or not self._rc_thread.is_alive():
            self._rc_thread = threading.Thread(
                target=self._rc_loop,
                name="tello-rc",
                daemon=True,
            )
            self._rc_thread.start()

        if self._wifi_thread is None or not self._wifi_thread.is_alive():
            self._wifi_thread = threading.Thread(
                target=self._wifi_loop,
                name="tello-wifi",
                daemon=True,
            )
            self._wifi_thread.start()

    def _telemetry_loop(self) -> None:
        # djitellopy reads telemetry from a state UDP packet the drone
        # broadcasts ~10 Hz. The `get_*` accessors read from that cache and
        # do NOT send commands, so polling is cheap. Link-health metrics
        # (packet loss, video errors, SNR, RTT) are merged in on top.
        while not self._closed:
            time.sleep(0.2)  # 5 Hz
            try:
                tele = self._read_telemetry()
            except Exception as exc:
                logger.debug("telemetry read failed: %s", exc)
                continue
            link = self._compute_link()
            with self._lock:
                self._last_telemetry = {**tele, **link}

    def _wifi_loop(self) -> None:
        # Poll the SDK ``wifi?`` command at 1 Hz. Each poll yields both the
        # WiFi SNR in dB and the round-trip latency of a command response,
        # which together are the cleanest "is the link still healthy"
        # signal the original Tello exposes.
        while not self._closed:
            time.sleep(1.0)
            if not self._connected:
                continue
            try:
                t0 = time.monotonic()
                raw = self._tello.query_wifi_signal_noise_ratio()
                rtt_ms = (time.monotonic() - t0) * 1000.0
                # Response is usually a bare integer like "90"; some
                # firmware revisions append "\r\n" or a unit. Be permissive.
                snr_str = "".join(c for c in str(raw) if c.isdigit() or c == "-")
                snr_db = int(snr_str) if snr_str else None
            except Exception as exc:
                logger.debug("wifi? query failed: %s", exc)
                # Don't clear the last value — let the staleness check in
                # ``_compute_link`` decide whether to ignore it. RTT we do
                # bump, so the dashboard reflects that the channel stalled.
                with self._link_lock:
                    self._link_rtt_ms = None
                continue
            with self._link_lock:
                self._link_snr_db = snr_db
                self._link_snr_ts = time.monotonic()
                self._link_rtt_ms = rtt_ms

    def _compute_link(self) -> dict[str, Any]:
        """Snapshot the link-health counters and compute the rates the
        dashboard cares about: packet loss %, video errors / sec, age of
        the last state packet, plus the most recent SNR + RTT readings.
        """

        now = time.monotonic()
        with _link_counters_lock:
            packets_total = _state_packets_total
            video_errors_total = _video_errors_total
            last_state_ts = _last_state_packet_mono

        self._link_history.append((now, packets_total, video_errors_total))

        packet_loss_pct: Optional[float] = None
        video_errors_per_sec: Optional[float] = None
        if len(self._link_history) >= 2:
            t0, p0, v0 = self._link_history[0]
            dt = now - t0
            if dt > 0:
                # Tello broadcasts state at ~10 Hz; missed packets translate
                # directly into observed rate < 10/s.
                pkt_rate = (packets_total - p0) / dt
                packet_loss_pct = max(0.0, min(100.0, (10.0 - pkt_rate) * 10.0))
                video_errors_per_sec = max(0.0, (video_errors_total - v0) / dt)

        ms_since_state = (
            (now - last_state_ts) * 1000.0 if last_state_ts > 0 else None
        )

        with self._link_lock:
            snr_db = self._link_snr_db
            snr_ts = self._link_snr_ts
            rtt_ms = self._link_rtt_ms

        # A stale SNR reading (no successful wifi? in the last 5 s) is more
        # misleading than no reading at all — the link is likely down. Drop
        # it so the dashboard / fence don't reason off a number that's no
        # longer true.
        snr_age_ms = (now - snr_ts) * 1000.0 if snr_ts > 0 else None
        if snr_age_ms is not None and snr_age_ms > 5000:
            snr_db = None

        return {
            "wifi_snr_db": snr_db,
            "link_rtt_ms": round(rtt_ms, 1) if rtt_ms is not None else None,
            "packet_loss_pct": (
                round(packet_loss_pct, 1) if packet_loss_pct is not None else None
            ),
            "video_errors_per_sec": (
                round(video_errors_per_sec, 1)
                if video_errors_per_sec is not None
                else None
            ),
            "ms_since_state": (
                round(ms_since_state) if ms_since_state is not None else None
            ),
        }

    def _rc_loop(self) -> None:
        # Continuously forward the current velocity vector to the drone while
        # in flight. Tello firmware auto-hovers if it stops receiving rc for
        # ~500 ms, so we have to be the heartbeat.
        period = 1.0 / RC_LOOP_HZ
        while not self._closed:
            time.sleep(period)
            with self._lock:
                if not self._flying:
                    continue
                lr, fb, ud, yaw = self._velocity
            try:
                self._tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as exc:
                logger.debug("send_rc_control failed: %s", exc)

    def _read_telemetry(self) -> dict[str, Any]:
        t = self._tello
        return {
            "battery_pct": t.get_battery(),
            "height_cm": t.get_height(),
            "tof_cm": t.get_distance_tof(),
            "flight_time_s": t.get_flight_time(),
            "temperature_c": t.get_temperature(),
            "barometer_m": t.get_barometer(),
            "pitch_deg": t.get_pitch(),
            "roll_deg": t.get_roll(),
            "yaw_deg": t.get_yaw(),
            "speed_x": t.get_speed_x(),
            "speed_y": t.get_speed_y(),
            "speed_z": t.get_speed_z(),
        }
