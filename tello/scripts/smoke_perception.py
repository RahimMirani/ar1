"""Smoke test for ``tello/perception.py`` — proves MiDaS loads + the depth
check runs on a static image without needing a drone.

Usage::

    uv run python tello/scripts/smoke_perception.py --image fixture.jpg
    uv run python tello/scripts/smoke_perception.py --image fixture.jpg --direction left

First run downloads the MiDaS Small ONNX (~21 MB) into
``~/.cache/firedrone/models/midas_small.onnx``. Subsequent runs are fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception import check_path_clear  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to JPEG/PNG to evaluate")
    ap.add_argument(
        "--direction",
        default="forward",
        choices=["forward", "left", "right", "up", "down"],
    )
    ap.add_argument("--near", type=float, default=0.55)
    ap.add_argument("--max-ratio", type=float, default=0.20)
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[FAIL] could not read image: {args.image}")
        return 1

    result = check_path_clear(
        frame,
        direction=args.direction,
        near_threshold=args.near,
        obstacle_max_ratio=args.max_ratio,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.available else 2


if __name__ == "__main__":
    sys.exit(main())
