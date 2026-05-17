"""Smoke test for ``tello/audio.py`` — proves the FFT detector picks up
a 3 kHz tone. No drone required.

Usage::

    # Just listen for 10 s and print live alarm-band / broadband dB:
    uv run python tello/scripts/smoke_audio.py

    # Listen for 30 s and a custom band floor:
    uv run python tello/scripts/smoke_audio.py --duration 30

Tip: play a smoke-alarm recording on your phone (or YouTube "smoke
alarm 3 khz") next to the mic to verify the detector fires.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from events import bus  # noqa: E402
from audio import monitor  # noqa: E402


async def main(duration: float) -> int:
    bus.attach_loop(asyncio.get_running_loop())
    q = bus.subscribe()

    status = monitor.start()
    if status.state == "error":
        print(f"[FAIL] mic open failed: {status.error}")
        return 1

    print(f"listening on {status.device or 'default device'} for {duration:.0f} s")
    print("event types: audio_level (per ~200 ms) + audio_alarm (state changes)")
    deadline = time.monotonic() + duration

    try:
        while time.monotonic() < deadline:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            t = ev.get("type")
            if t == "audio_level":
                tonality = ev.get("tonality_db")
                broad    = ev.get("broadband_db")
                peak     = ev.get("peak_freq_hz")
                pulses   = ev.get("pulses_recent")
                state    = ev.get("state")
                print(
                    f"level  tonality={tonality:+6.1f} dB  "
                    f"peak={peak:7.1f} Hz  pulses={pulses}  "
                    f"broad={broad:+6.1f} dB  state={state}",
                    flush=True,
                )
            elif t == "audio_alarm":
                print(f"ALARM  {ev}", flush=True)
    finally:
        monitor.stop()
        bus.unsubscribe(q)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to listen")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.duration)))
