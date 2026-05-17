"""Standalone smoke test for the 2D mapping layer (``tello/mapping.py``).

Runs without a physical Tello or any network access. Feeds a synthetic
telemetry stream into the ``Mapper`` and asserts the integrated pose +
auxiliary state evolve the way they should:

  §1  pose integrates body-frame velocity into the world frame
  §2  yaw rotates the velocity correctly (turning then driving
      moves the drone perpendicular, not forward)
  §3  takeoff transition (flying False -> True) resets pose to origin
  §4  belly-cam lockout detector trips when velocity reads ~0 while
      the IMU sees acceleration, and the position integral freezes
      during that window
  §5  the renderer produces a non-trivial JPEG once the mapper has
      seen at least one tick of flight

The MiDaS-dependent obstacle stamping is not exercised here — that
requires a real image to feed the model. ``smoke_perception.py``
already covers MiDaS end-to-end; this script targets the integrator
+ confidence + render plumbing that's unique to ``mapping.py``.

Exits 0 on success, 1 on any failure. Designed to be cheap (under 10
seconds wall-clock) so we can run it on every edit to mapping.py
before flight testing.

Run:

    uv run python tello/scripts/smoke_mapping.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable

# Allow running from the repo root by adding tello/ to sys.path before
# importing mapping. Mirrors smoke_link's pattern.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TELLO = os.path.dirname(_HERE)
if _TELLO not in sys.path:
    sys.path.insert(0, _TELLO)

import mapping  # noqa: E402
from mapping import Mapper, POSE_HZ  # noqa: E402

import numpy as np  # noqa: E402


# --------------------------------------------------------------------------- #
# Tiny test harness — no pytest dependency.
# --------------------------------------------------------------------------- #

_FAILED: list[str] = []


def check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except AssertionError as exc:
        _FAILED.append(f"{name}: {exc}")
        print(f"  FAIL  {name} - {exc}")
    except Exception as exc:
        _FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  ERROR {name} - {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {name}")


# --------------------------------------------------------------------------- #
# Synthetic drone harness — same shape as the real Drone for the three
# accessors the Mapper consumes.
# --------------------------------------------------------------------------- #


class FakeDrone:
    def __init__(self) -> None:
        self.telemetry: dict[str, Any] = {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "speed_x": 0.0,
            "speed_y": 0.0,
            "speed_z": 0.0,
            "accel_x_mg": 0.0,
            "accel_y_mg": 0.0,
            "accel_z_mg": 1000.0,  # ~1 g down at rest
        }
        self.flying: bool = False
        self.frame: np.ndarray | None = None

    def get_telemetry(self) -> dict[str, Any]:
        return self.telemetry

    def get_frame(self) -> np.ndarray | None:
        return self.frame

    def get_flying(self) -> bool:
        return self.flying


def _make_mapper(drone: FakeDrone) -> Mapper:
    return Mapper(
        get_telemetry=drone.get_telemetry,
        get_frame=drone.get_frame,
        get_flying=drone.get_flying,
    )


def _wait_for_ticks(n: int) -> None:
    """Sleep just long enough for `n` pose-loop ticks to land. Padded
    because thread scheduling on Windows isn't exact."""
    time.sleep(n / POSE_HZ + 0.15)


# --------------------------------------------------------------------------- #
# §1 — forward velocity integrates into world +x
# --------------------------------------------------------------------------- #


def test_forward_velocity_integrates_world_x() -> None:
    drone = FakeDrone()
    m = _make_mapper(drone)
    m.start()
    try:
        drone.flying = True
        _wait_for_ticks(2)
        # Drive forward at 50 cm/s for ~1.5 s.
        drone.telemetry["speed_x"] = 50.0
        _wait_for_ticks(15)
        pose = m.snapshot().pose
        assert pose["x_m"] > 0.4, (
            f"forward velocity should move +x; got x={pose['x_m']:+.3f} m"
        )
        assert abs(pose["y_m"]) < 0.15, (
            f"pure forward velocity should leave y~0; got y={pose['y_m']:+.3f} m"
        )
    finally:
        m.stop()


# --------------------------------------------------------------------------- #
# §2 — yaw rotates the velocity vector
# --------------------------------------------------------------------------- #


def test_yaw_rotates_velocity() -> None:
    drone = FakeDrone()
    m = _make_mapper(drone)
    m.start()
    try:
        drone.flying = True
        _wait_for_ticks(2)
        # Face 90 degrees right (heading=90 deg) and drive forward.
        # With our convention (theta positive CW), this should advance
        # the drone along +y, not +x.
        drone.telemetry["yaw_deg"] = 90.0
        drone.telemetry["speed_x"] = 50.0
        _wait_for_ticks(15)
        pose = m.snapshot().pose
        # +x should stay near 0; +y should grow.
        assert pose["y_m"] > 0.4, (
            f"heading 90 deg + forward should advance +y; got y={pose['y_m']:+.3f} m"
        )
        assert abs(pose["x_m"]) < 0.15, (
            f"heading 90 deg + forward should leave x~0; got x={pose['x_m']:+.3f} m"
        )
    finally:
        m.stop()


# --------------------------------------------------------------------------- #
# §3 — takeoff transition resets pose
# --------------------------------------------------------------------------- #


def test_takeoff_transition_resets_pose() -> None:
    drone = FakeDrone()
    m = _make_mapper(drone)
    m.start()
    try:
        # Fly for a bit, accumulate some position.
        drone.flying = True
        drone.telemetry["speed_x"] = 50.0
        _wait_for_ticks(8)
        pose_mid = m.snapshot().pose
        assert pose_mid["x_m"] > 0.15, (
            f"prep: expected non-zero pose before reset, got x={pose_mid['x_m']:+.3f}"
        )
        # Land, then take off again — the False -> True transition
        # should reset pose to origin and start a fresh trajectory.
        drone.flying = False
        drone.telemetry["speed_x"] = 0.0
        _wait_for_ticks(2)
        drone.flying = True
        _wait_for_ticks(2)
        pose_after = m.snapshot().pose
        # Allow a tiny epsilon because the very next pose tick after
        # reset still integrates one dt.
        assert abs(pose_after["x_m"]) < 0.05 and abs(pose_after["y_m"]) < 0.05, (
            f"takeoff transition should reset pose; got x={pose_after['x_m']:+.3f}, "
            f"y={pose_after['y_m']:+.3f}"
        )
    finally:
        m.stop()


# --------------------------------------------------------------------------- #
# §4 — belly-cam lockout freezes integration + flips confidence to "low"
# --------------------------------------------------------------------------- #


def test_lockout_freezes_pose_and_flags_low_confidence() -> None:
    drone = FakeDrone()
    m = _make_mapper(drone)
    m.start()
    try:
        drone.flying = True
        _wait_for_ticks(2)
        # Cleanly cruise at 50 cm/s for ~1 s so we have some position.
        drone.telemetry["speed_x"] = 50.0
        drone.telemetry["accel_x_mg"] = 10.0
        _wait_for_ticks(10)
        pose_cruise = m.snapshot().pose
        assert pose_cruise["x_m"] > 0.3, "prep: expected forward progress"

        # Now simulate optical-flow lockout: velocity reads ~0 but the
        # IMU sees significant horizontal accel. Sustain for >=6 ticks
        # (= LOCKOUT_CONSECUTIVE_TICKS) so the detector commits.
        drone.telemetry["speed_x"] = 0.0
        drone.telemetry["accel_x_mg"] = 200.0  # well above LOCKOUT_ACCEL_MG=80
        _wait_for_ticks(12)
        snap = m.snapshot()
        assert snap.pose_confidence == "low", (
            f"sustained vel~0 + accel>80mg should trip lockout; "
            f"got confidence={snap.pose_confidence}"
        )
        # Position must NOT have continued to grow during lockout (the
        # integral is frozen). It may have crept up by less than one
        # tick worth before the detector latched, hence the lax bound.
        pose_lock = snap.pose
        assert pose_lock["x_m"] - pose_cruise["x_m"] < 0.10, (
            f"lockout should freeze the integral; "
            f"x crept from {pose_cruise['x_m']:+.3f} to {pose_lock['x_m']:+.3f}"
        )

        # Recover: accel drops, velocity returns to normal. After a
        # few ticks confidence should swing back to ok.
        drone.telemetry["accel_x_mg"] = 0.0
        drone.telemetry["speed_x"] = 0.0
        _wait_for_ticks(12)
        recovered = m.snapshot().pose_confidence
        assert recovered == "ok", (
            f"lockout should clear once accel + vel disagree no more; got {recovered}"
        )
    finally:
        m.stop()


# --------------------------------------------------------------------------- #
# §5 — renderer produces a non-trivial JPEG
# --------------------------------------------------------------------------- #


def test_render_emits_jpeg_after_first_tick() -> None:
    drone = FakeDrone()
    m = _make_mapper(drone)
    m.start()
    try:
        drone.flying = True
        drone.telemetry["speed_x"] = 30.0
        # Give the render thread (2 Hz) at least one cycle to fire.
        time.sleep(1.0)
        payload = m.latest_jpeg()
        assert payload is not None, "render loop should produce a JPEG within 1 s"
        # JPEGs start with the FF D8 FF magic. We don't decode it — we
        # just confirm the bytes look like a real image, not zeros.
        assert len(payload) > 400, (
            f"rendered JPEG looks too small to be a real image: {len(payload)} bytes"
        )
        assert payload[:3] == b"\xff\xd8\xff", (
            f"rendered payload is not a JPEG (magic={payload[:3].hex()})"
        )
    finally:
        m.stop()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    print("smoke_mapping - 2D pose + map regression checks\n")
    check("§1  forward velocity integrates world +x",       test_forward_velocity_integrates_world_x)
    check("§2  yaw rotates body velocity into world",       test_yaw_rotates_velocity)
    check("§3  takeoff transition resets pose",             test_takeoff_transition_resets_pose)
    check("§4  flow lockout freezes pose + flags low conf", test_lockout_freezes_pose_and_flags_low_confidence)
    check("§5  renderer emits JPEG within 1 s",             test_render_emits_jpeg_after_first_tick)

    print()
    if _FAILED:
        print(f"FAILED - {len(_FAILED)} check(s):")
        for f in _FAILED:
            print(f"  - {f}")
        return 1
    print("All checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
