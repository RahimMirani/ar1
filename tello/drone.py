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
from dataclasses import dataclass, field
from typing import Any, Optional

import av as _av
from djitellopy import Tello

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Low-latency video decoder
#
# djitellopy opens its H.264 UDP stream via PyAV's ``av.open`` and does not
# expose any way to pass decoder options. Out of the box libav buffers several
# frames for B-frame reordering and "smooth playback", which adds 100-300 ms
# of latency to the dashboard video. We monkey-patch ``av.open`` exactly once
# at import time to inject low-delay options whenever the URL is a UDP stream.
# Non-UDP callers are unaffected.
# --------------------------------------------------------------------------- #

_LOW_LATENCY_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "32",
    "analyzeduration": "0",
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
        """Return the latest decoded frame as a numpy array, or None."""
        if not self._streaming:
            return None
        try:
            return self._tello.get_frame_read().frame
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

    def _telemetry_loop(self) -> None:
        # djitellopy reads telemetry from a state UDP packet the drone
        # broadcasts ~10 Hz. The `get_*` accessors read from that cache and
        # do NOT send commands, so polling is cheap.
        while not self._closed:
            time.sleep(0.2)  # 5 Hz
            try:
                tele = self._read_telemetry()
            except Exception as exc:
                logger.debug("telemetry read failed: %s", exc)
                continue
            with self._lock:
                self._last_telemetry = tele

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
