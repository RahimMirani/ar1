"""Smoke test: print live NimbusOS telemetry for a few seconds.

Run with the drone powered on and the NimbusOS desktop app running:

    uv run firedrone-telemetry

If nothing prints, the SDK / dongle / drone path is broken and the rest of
the project will not work. Fix that before going further.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from nimbusos_sdk import NimbusClient


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="How long to listen for telemetry before exiting (default: 5s).",
    )
    args = parser.parse_args()

    print(f"Listening for telemetry for ~{args.seconds:.1f}s ...", flush=True)

    received = 0
    with NimbusClient() as client:
        for telemetry in client.telemetry(timeout_sec=args.seconds):
            received += 1
            battery = telemetry.battery
            attitude = telemetry.attitude
            print(
                f"#{received:03d}  "
                f"batt={battery.voltage:5.2f}V  "
                f"roll={attitude.roll_deg:6.1f}  "
                f"pitch={attitude.pitch_deg:6.1f}  "
                f"yaw={attitude.yaw_deg:6.1f}",
                flush=True,
            )

    if received == 0:
        print(
            "ERROR: no telemetry received. Is NimbusOS running and the drone on?",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"OK: received {received} telemetry messages.")
    sys.exit(0)


if __name__ == "__main__":
    main()
