"""Smoke test for ``tello/vision.py`` — proves the OpenAI vision path end to end.

Two modes:

* ``--image PATH`` analyzes a static JPEG/PNG on disk. No drone needed.
  Use this as the first sanity check after wiring up ``OPENAI_API_KEY``.

* No arguments: grabs a live frame from the Tello, analyzes it, and
  prints the result. Needs the laptop joined to ``TELLO-XXXXXX`` WiFi.

Run::

    uv run python tello/scripts/smoke_vision.py --image fixture.jpg
    uv run python tello/scripts/smoke_vision.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from vision import analyze_frame  # noqa: E402


def grab_live_frame():
    from drone import Drone  # local import — keeps --image path drone-free

    d = Drone()
    print("Connecting to Tello at 192.168.10.1 ...")
    d.connect()
    d.start_stream()
    for _ in range(30):
        frame = d.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.1)
    raise RuntimeError("no frame from drone within 3 s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to a local JPEG/PNG instead of live frame")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[FAIL] could not read image: {args.image}")
            sys.exit(1)
    else:
        try:
            frame = grab_live_frame()
        except Exception as exc:
            print(f"[FAIL] no live frame ({exc}). Try --image PATH.")
            sys.exit(1)

    try:
        result = analyze_frame(frame)
    except Exception as exc:
        print(f"[FAIL] vision call failed: {exc}")
        sys.exit(1)

    out = result.to_dict()
    # The thumbnail is a few KB of base64 — too noisy for stdout.
    out.pop("thumbnail_b64", None)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
