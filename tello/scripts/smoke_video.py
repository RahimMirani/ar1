"""Smoke test: open an OpenCV window with the live Tello camera feed.

No flight. Run this second, after `smoke_telemetry.py` succeeds, to confirm
the video pipeline works end-to-end.

Press `q` in the window to quit.

Usage::

    cd tello
    uv run python scripts/smoke_video.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone import Drone  # noqa: E402

import cv2


WINDOW_TITLE = "Tello video (q to quit)"


def main() -> int:
    print("Connecting to Tello at 192.168.10.1 ...")
    try:
        drone = Drone()
        drone.connect()
        drone.start_stream()
    except Exception as exc:
        print(f"[FAIL] could not start stream: {exc}")
        return 1

    print("Stream on. Opening OpenCV window. Press 'q' to quit.")

    try:
        # Give the first frame a moment to arrive.
        time.sleep(1.0)
        while True:
            frame = drone.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        drone.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
