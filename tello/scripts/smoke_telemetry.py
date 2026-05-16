"""Smoke test: connect to the Tello and print telemetry for ~5 seconds.

No flight, no video. Run this first to confirm the laptop is on the Tello WiFi
and the SDK chain works.

Usage::

    cd tello
    uv run python scripts/smoke_telemetry.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone import Drone  # noqa: E402


DURATION_S = 5.0
POLL_HZ = 2.0


def main() -> int:
    print("Connecting to Tello at 192.168.10.1 ...")
    try:
        drone = Drone()
        drone.connect()
    except Exception as exc:
        print(f"[FAIL] could not connect to Tello: {exc}")
        print("       Is your laptop joined to the TELLO-XXXXXX WiFi?")
        return 1

    print(f"Connected. Streaming telemetry for {DURATION_S:.1f}s ...\n")

    try:
        end = time.monotonic() + DURATION_S
        period = 1.0 / POLL_HZ
        while time.monotonic() < end:
            snap = drone.snapshot()
            t = snap.telemetry
            print(
                f"batt={t.get('battery_pct')}%  "
                f"h={t.get('height_cm')}cm  "
                f"tof={t.get('tof_cm')}cm  "
                f"yaw={t.get('yaw_deg')}deg  "
                f"pitch={t.get('pitch_deg')}deg  "
                f"roll={t.get('roll_deg')}deg  "
                f"temp={t.get('temperature_c')}C"
            )
            time.sleep(period)
    finally:
        drone.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
