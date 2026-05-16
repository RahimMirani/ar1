"""High-level wrapper around `nimbusos_sdk.NimbusClient`.

This is the only module in the project that is allowed to make raw SDK calls.
Everything else (teleop, the future audio / vision / agent code) should use
the :class:`Drone` class so the safety, state caching, and coordinate-frame
conversions live in one place.

Conventions
-----------
* The NimbusOS local frame uses ``forward / right / down``. ``down`` is the
  z-axis target in meters: **negative is up**.
* Public methods take a friendly ``altitude_m`` (positive = up) and convert
  to ``down`` internally so callers never have to think about sign flips.
* Position commands are clamped to a small indoor envelope so a stray key
  press or buggy caller cannot send the drone across the room.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Optional

from nimbusos_sdk import NimbusClient, State, Telemetry

# Indoor envelope. Tune in `drone.py` only; do not pass overrides from teleop.
MAX_FORWARD_M = 2.0
MAX_RIGHT_M = 2.0
MIN_ALTITUDE_M = 0.3
MAX_ALTITUDE_M = 2.0

# Safety thresholds. The watchdog thread enforces these.
MAX_FLIGHT_SECONDS = 90.0
LOW_BATTERY_VOLTS = 3.4
LOW_BATTERY_CONSECUTIVE = 3  # debounce noisy readings

# Default takeoff altitude.
DEFAULT_TAKEOFF_ALTITUDE_M = 1.0

# How close to the ground we need to be to call a landing "done".
LANDING_Z_TOLERANCE_M = 0.15
LANDING_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class DroneSnapshot:
    """Immutable read of the latest cached telemetry + state."""

    armed: bool
    flying: bool
    battery_volts: Optional[float]
    yaw_deg: Optional[float]
    x_m: Optional[float]
    y_m: Optional[float]
    z_m: Optional[float]
    target_forward_m: float
    target_right_m: float
    target_down_m: float
    flight_seconds: float
    last_status: str


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class Drone:
    """Semantic wrapper around the NimbusOS SDK.

    Usage::

        with Drone() as drone:
            drone.takeoff()
            drone.nudge(d_forward=0.2)
            drone.land_and_disarm()
    """

    def __init__(
        self,
        *,
        pub_endpoint: str = "tcp://127.0.0.1:7771",
        sub_endpoint: str = "tcp://127.0.0.1:7772",
    ) -> None:
        self._client = NimbusClient(pub_endpoint=pub_endpoint, sub_endpoint=sub_endpoint)

        self._lock = threading.RLock()
        self._closed = False

        self._latest_telemetry: Optional[Telemetry] = None
        self._latest_state: Optional[State] = None
        self._low_battery_hits = 0

        # Commanded target in the local frame. We re-publish this on every
        # `nudge` / `go_to` / `hover` so the flight controller keeps moving
        # toward the latest desired position.
        self._target_forward = 0.0
        self._target_right = 0.0
        self._target_down = 0.0  # 0 == ground

        self._armed = False
        self._flying = False
        self._arm_time: Optional[float] = None
        self._last_status = "DISARMED"

        # Background threads. Started in `__enter__`.
        self._telemetry_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "Drone":
        self._start_background_threads()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        """Disarm (best-effort) and shut down background threads + client."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._armed:
                # Don't try to land - we may not be flying. Just disarm.
                self._safe_publish_arm(False)
        finally:
            self._client.close()

    # ------------------------------------------------------------------ #
    # Public commands
    # ------------------------------------------------------------------ #

    def arm(self) -> None:
        with self._lock:
            self._client.publish_arm_state(True)
            self._armed = True
            self._arm_time = time.monotonic()
            self._last_status = "ARMED"

    def disarm(self) -> None:
        with self._lock:
            self._safe_publish_arm(False)
            self._armed = False
            self._flying = False
            self._arm_time = None
            self._last_status = "DISARMED"

    def takeoff(self, altitude_m: float = DEFAULT_TAKEOFF_ALTITUDE_M) -> None:
        """Arm, enter guided mode, and rise to ``altitude_m`` above start."""
        altitude_m = _clamp(altitude_m, MIN_ALTITUDE_M, MAX_ALTITUDE_M)
        self.arm()
        time.sleep(0.3)
        self._client.publish_guidance_request("go")
        time.sleep(0.3)
        with self._lock:
            self._target_forward = 0.0
            self._target_right = 0.0
            self._target_down = -altitude_m
            self._flying = True
            self._last_status = f"TAKEOFF -> {altitude_m:.2f}m"
        self._send_target_unlocked()

    def nudge(
        self,
        *,
        d_forward: float = 0.0,
        d_right: float = 0.0,
        d_down: float = 0.0,
    ) -> None:
        """Move the commanded target by a delta and re-publish.

        ``d_down`` follows the NED convention: positive moves the target
        toward the ground. Use negative deltas to climb.
        """
        with self._lock:
            self._target_forward += d_forward
            self._target_right += d_right
            self._target_down += d_down
            self._last_status = "MOVING"
            self._send_target_unlocked()

    def go_to(self, forward: float, right: float, altitude_m: float) -> None:
        """Set an absolute target waypoint in the local frame."""
        with self._lock:
            self._target_forward = forward
            self._target_right = right
            self._target_down = -altitude_m
            self._last_status = (
                f"GO_TO f={forward:.2f} r={right:.2f} alt={altitude_m:.2f}"
            )
            self._send_target_unlocked()

    def rotate(self, radians: float) -> None:
        """Yaw by ``radians`` (positive = right / clockwise from above)."""
        self._client.publish_yaw_turn_command(radians)
        with self._lock:
            self._last_status = f"YAW {radians:+.2f} rad"

    def hover(self) -> None:
        """Cancel any in-progress motion by re-sending the current target."""
        with self._lock:
            self._last_status = "HOVER"
            self._send_target_unlocked()

    def land_and_disarm(self) -> None:
        """Request a controlled landing and disarm once on the ground."""
        with self._lock:
            self._last_status = "LANDING"
        self._client.publish_guidance_request("land")

        deadline = time.monotonic() + LANDING_TIMEOUT_S
        while time.monotonic() < deadline:
            z = self._cached_z_m()
            if z is not None and abs(z) <= LANDING_Z_TOLERANCE_M:
                break
            time.sleep(0.2)

        with self._lock:
            self._safe_publish_arm(False)
            self._armed = False
            self._flying = False
            self._arm_time = None
            self._last_status = "LANDED"

    def emergency_land(self) -> None:
        """Best-effort land + disarm. Never raises."""
        try:
            self._client.publish_guidance_request("land")
        except Exception:
            pass
        time.sleep(0.5)
        with self._lock:
            self._safe_publish_arm(False)
            self._armed = False
            self._flying = False
            self._arm_time = None
            self._last_status = "EMERGENCY LANDED"

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def snapshot(self) -> DroneSnapshot:
        with self._lock:
            tele = self._latest_telemetry
            state = self._latest_state
            flight_seconds = (
                time.monotonic() - self._arm_time if self._arm_time is not None else 0.0
            )
            x, y, z = None, None, None
            if state is not None and state.valid:
                x = state.position.x_m
                y = state.position.y_m
                z = state.position.z_m
            return DroneSnapshot(
                armed=self._armed,
                flying=self._flying,
                battery_volts=(tele.battery.voltage if tele is not None else None),
                yaw_deg=(tele.attitude.yaw_deg if tele is not None else None),
                x_m=x,
                y_m=y,
                z_m=z,
                target_forward_m=self._target_forward,
                target_right_m=self._target_right,
                target_down_m=self._target_down,
                flight_seconds=flight_seconds,
                last_status=self._last_status,
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _send_target_unlocked(self) -> None:
        # Caller must hold self._lock.
        f = _clamp(self._target_forward, -MAX_FORWARD_M, MAX_FORWARD_M)
        r = _clamp(self._target_right, -MAX_RIGHT_M, MAX_RIGHT_M)
        d = _clamp(self._target_down, -MAX_ALTITUDE_M, -MIN_ALTITUDE_M)

        self._target_forward = f
        self._target_right = r
        self._target_down = d

        self._client.publish_waypoint_command(
            forward=f,
            right=r,
            down=d,
            mode="override",
            threshold_m=0.15,
            hold_time_s=0.0,
        )

    def _safe_publish_arm(self, armed: bool) -> None:
        try:
            self._client.publish_arm_state(armed)
        except Exception:
            pass

    def _cached_z_m(self) -> Optional[float]:
        with self._lock:
            state = self._latest_state
        if state is not None and state.valid:
            return state.position.z_m
        return None

    def _start_background_threads(self) -> None:
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="firedrone-telemetry",
            daemon=True,
        )
        self._state_thread = threading.Thread(
            target=self._state_loop,
            name="firedrone-state",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="firedrone-watchdog",
            daemon=True,
        )
        self._telemetry_thread.start()
        self._state_thread.start()
        self._watchdog_thread.start()

    def _telemetry_loop(self) -> None:
        while not self._closed:
            try:
                for tele in self._client.telemetry():
                    if self._closed:
                        return
                    with self._lock:
                        self._latest_telemetry = tele
                    self._check_battery(tele.battery.voltage)
            except Exception:
                time.sleep(0.5)

    def _state_loop(self) -> None:
        while not self._closed:
            try:
                for state in self._client.state():
                    if self._closed:
                        return
                    with self._lock:
                        self._latest_state = state
            except Exception:
                time.sleep(0.5)

    def _watchdog_loop(self) -> None:
        while not self._closed:
            time.sleep(1.0)
            with self._lock:
                if not self._armed or self._arm_time is None:
                    continue
                flight_seconds = time.monotonic() - self._arm_time
                expired = flight_seconds > MAX_FLIGHT_SECONDS
            if expired:
                self._last_status = "WATCHDOG: max flight time -> emergency land"
                self.emergency_land()

    def _check_battery(self, volts: float) -> None:
        if volts is None:
            return
        with self._lock:
            if volts < LOW_BATTERY_VOLTS:
                self._low_battery_hits += 1
            else:
                self._low_battery_hits = 0
            tripped = (
                self._armed and self._low_battery_hits >= LOW_BATTERY_CONSECUTIVE
            )
        if tripped:
            self._last_status = f"LOW BATTERY {volts:.2f}V -> land"
            try:
                self.land_and_disarm()
            except Exception:
                self.emergency_land()
