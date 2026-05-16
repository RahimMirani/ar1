"""High-level wrapper around `djitellopy.Tello`.

The only module in the `tello/` project allowed to make raw SDK calls. The
FastAPI server, smoke scripts, and (later) the agent code all go through the
:class:`Drone` class so connection state, telemetry caching, and unit
conversion live in one place.

Conventions
-----------
* Movement distances are in **centimeters** (Tello SDK native). Valid range
  is 20-500 cm per command.
* Yaw is in **degrees**, 1-360 per command.
* Flips take a single character: ``"f"`` forward, ``"b"`` back, ``"l"`` left,
  ``"r"`` right.
* No flight-time cap and no low-battery auto-land in milestone 1 — only the
  WebSocket-disconnect emergency stop in `main.py` provides automatic safety.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from djitellopy import Tello

logger = logging.getLogger(__name__)

# Per-command movement bounds in cm. The Tello SDK rejects values outside this.
MIN_MOVE_CM = 20
MAX_MOVE_CM = 500

# Per-command yaw bounds in degrees.
MIN_YAW_DEG = 1
MAX_YAW_DEG = 360

# Step sizes used by the dashboard's keyboard/button controls.
DEFAULT_STEP_CM = 30
DEFAULT_YAW_DEG = 30

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
            drone.move("forward", 50)
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

        self._telemetry_thread: Optional[threading.Thread] = None

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
        self._start_telemetry_loop()

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
    # Flight commands
    # ------------------------------------------------------------------ #

    def takeoff(self) -> None:
        with self._lock:
            self._tello.takeoff()
            self._flying = True
            self._last_status = "TAKEOFF"

    def land(self) -> None:
        with self._lock:
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
            self._flying = False
            self._last_status = "EMERGENCY"

    def move(self, direction: str, distance_cm: int = DEFAULT_STEP_CM) -> None:
        """Move in the named direction by ``distance_cm`` (clamped to 20-500)."""
        if direction not in VALID_MOVE_DIRECTIONS:
            raise ValueError(f"Invalid move direction: {direction!r}")
        distance_cm = _clamp_int(distance_cm, MIN_MOVE_CM, MAX_MOVE_CM)
        method_name = f"move_{direction}"
        method = getattr(self._tello, method_name)
        with self._lock:
            method(distance_cm)
            self._last_status = f"MOVE {direction.upper()} {distance_cm}cm"

    def rotate(self, direction: str, degrees: int = DEFAULT_YAW_DEG) -> None:
        """Yaw ``cw`` (right) or ``ccw`` (left) by ``degrees`` (1-360)."""
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

    def flip(self, direction: str) -> None:
        """Flip in one of the four cardinal directions. Needs battery > 50%."""
        if direction not in VALID_FLIP_DIRECTIONS:
            raise ValueError(f"Invalid flip direction: {direction!r}")
        with self._lock:
            self._tello.flip(direction)
            self._last_status = f"FLIP {direction.upper()}"

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def snapshot(self) -> DroneSnapshot:
        with self._lock:
            return DroneSnapshot(
                connected=self._connected,
                streaming=self._streaming,
                flying=self._flying,
                last_status=self._last_status,
                last_error=self._last_error,
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

    def _start_telemetry_loop(self) -> None:
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            return
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="tello-telemetry",
            daemon=True,
        )
        self._telemetry_thread.start()

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
