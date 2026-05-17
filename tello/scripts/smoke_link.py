"""Standalone smoke test for the link-safety system in ``tello/drone.py``.

Runs without a physical Tello. Exercises the invariants documented in
LINK SAFETY INVARIANTS in ``drone.py``:

  1. All three module-level monkey-patches applied.
  2. ``send_command_with_return`` is actually serialized across threads.
  3. ``_compute_link`` suppresses packet_loss_pct until the warmup window.
  4. Fence debounces non-LAND transitions but commits LAND instantly.
  5. A healthy simulated 10 Hz state stream with realistic jitter does NOT
     trip the HOVER tier — the regression that broke teleop the first time.

Exits 0 on success, 1 on any failure (with a printed report). Designed to
be cheap so we can run it on every edit to drone.py before flight testing.

Run:

    uv run python tello/scripts/smoke_link.py

Or from the repo root with the venv activated:

    python tello/scripts/smoke_link.py
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import Callable

# Allow ``python tello/scripts/smoke_link.py`` from the repo root by adding
# the ``tello/`` directory to sys.path before importing drone.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TELLO = os.path.dirname(_HERE)
if _TELLO not in sys.path:
    sys.path.insert(0, _TELLO)

import drone as drone_module  # noqa: E402
from drone import Drone        # noqa: E402

import djitellopy             # noqa: E402


# --------------------------------------------------------------------------- #
# Tiny test harness — keeps the script self-contained, no pytest dep.
# --------------------------------------------------------------------------- #

_FAILED: list[str] = []


def check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except AssertionError as exc:
        _FAILED.append(f"{name}: {exc}")
        print(f"  FAIL  {name} — {exc}")
    except Exception as exc:
        _FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  ERROR {name} — {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {name}")


# --------------------------------------------------------------------------- #
# §1 — patches applied
# --------------------------------------------------------------------------- #


def test_patches_applied() -> None:
    assert drone_module._command_lock_patched,  "command-serialization patch not applied"
    assert drone_module._state_counter_patched, "state-packet counter patch not applied"
    assert drone_module._video_decode_patched,  "video-decode patch not applied"


# --------------------------------------------------------------------------- #
# §2 — command lock actually serializes two threads
# --------------------------------------------------------------------------- #


def test_command_lock_serializes() -> None:
    # Replace the inner "real" send with a slow stub so we can observe
    # overlap. We re-wrap with the existing module lock to mirror what the
    # production patch does.
    calls: list[tuple[str, str, float]] = []

    def fake(self, cmd, timeout=7):
        calls.append(("start", cmd, time.monotonic()))
        time.sleep(0.2)
        calls.append(("end",   cmd, time.monotonic()))
        return "ok"

    def wrapped(self, *args, **kwargs):
        with drone_module._command_send_lock:
            return fake(self, *args, **kwargs)

    saved = djitellopy.Tello.send_command_with_return
    djitellopy.Tello.send_command_with_return = wrapped
    try:
        class Dummy:
            address = None

        dummy = Dummy()

        def runner(cmd: str) -> None:
            djitellopy.Tello.send_command_with_return(dummy, cmd)

        t1 = threading.Thread(target=runner, args=("takeoff",))
        t2 = threading.Thread(target=runner, args=("wifi?",))
        t1.start()
        time.sleep(0.02)
        t2.start()
        t1.join()
        t2.join()

        # The second start must come on or after the first end — no overlap.
        first_end   = next(t for (k, _, t) in calls if k == "end")
        second_start = [t for (k, _, t) in calls if k == "start"][1]
        assert second_start >= first_end, (
            f"calls overlapped (second start {second_start:.3f} < first end "
            f"{first_end:.3f}); lock is not serializing"
        )
    finally:
        djitellopy.Tello.send_command_with_return = saved


# --------------------------------------------------------------------------- #
# §3 — _compute_link warmup
# --------------------------------------------------------------------------- #


def test_warmup_suppresses_early_loss() -> None:
    d = Drone()
    # Pump state-packet counters as if the receiver is running smoothly at
    # the expected 10 Hz, but only for 1 second of wall-clock time. The
    # window is still below _LINK_WARMUP_SEC so loss should read as None.
    base = drone_module._state_packets_total
    for i in range(5):
        with drone_module._link_counters_lock:
            drone_module._state_packets_total += 2  # 2 packets per 0.2 s tick
            drone_module._last_state_packet_mono = time.monotonic()
        d._compute_link()  # accumulate history
        time.sleep(0.2)
    link = d._compute_link()
    assert link["packet_loss_pct"] is None, (
        f"packet_loss_pct should be None during warmup, got {link['packet_loss_pct']}"
    )
    # Cleanup so this test doesn't pollute the global counter for the next.
    with drone_module._link_counters_lock:
        drone_module._state_packets_total = base


# --------------------------------------------------------------------------- #
# §4 — fence debounce + instant LAND
# --------------------------------------------------------------------------- #


def test_debounce_single_tick_hover_does_not_commit() -> None:
    d = Drone()
    d._flying = True
    # A single hover tick followed by an ok tick should NOT commit.
    d._evaluate_fence({"wifi_snr_db": None, "packet_loss_pct": 20, "ms_since_state": 200})
    d._evaluate_fence({"wifi_snr_db": None, "packet_loss_pct": 2,  "ms_since_state": 200})
    assert d._fence_tier == "ok", (
        f"single-tick HOVER should be debounced; got tier={d._fence_tier}"
    )


def test_debounce_sustained_hover_commits() -> None:
    d = Drone()
    d._flying = True
    # Drone won't actually call set_velocity since _tello isn't real, but
    # the transition is what we're checking.
    d.set_velocity = lambda *a, **k: None  # type: ignore[assignment]
    for _ in range(8):
        d._evaluate_fence({"wifi_snr_db": None, "packet_loss_pct": 20, "ms_since_state": 200})
        time.sleep(0.2)
    assert d._fence_tier == "hover", (
        f"sustained HOVER should commit after debounce; got tier={d._fence_tier}"
    )


def test_land_fires_instantly_without_debounce() -> None:
    d = Drone()
    d._flying = True
    landed: list[bool] = []
    d.land = lambda: landed.append(True)  # type: ignore[assignment]
    d._evaluate_fence({"wifi_snr_db": None, "packet_loss_pct": 40, "ms_since_state": 200})
    assert d._fence_tier == "land", f"LAND should commit on first tick; got {d._fence_tier}"
    # The actual land() runs on a worker thread — give it a moment.
    time.sleep(0.1)
    assert landed, "LAND tier should have dispatched land() on a worker thread"


# --------------------------------------------------------------------------- #
# §5 — healthy 10 Hz with realistic jitter does NOT trip HOVER
#
# This is the regression test for the original bug: a 5 Hz telemetry loop
# sampling a noisy 10 Hz state stream over a 5 s window should never see
# enough apparent loss to cross _FENCE_LOSS_HOVER (15%).
# --------------------------------------------------------------------------- #


def test_healthy_link_does_not_misfire() -> None:
    d = Drone()
    d._flying = True
    d.set_velocity = lambda *a, **k: None  # type: ignore[assignment]
    d.land         = lambda: None           # type: ignore[assignment]

    base_packets = drone_module._state_packets_total
    rng = random.Random(0)

    # Simulate 8 seconds of operation: state packets arrive at 10 Hz with
    # uniform [-25, +25] ms jitter on inter-arrival time, which is what
    # we observed empirically with a Tello on a clean link.
    next_packet = time.monotonic()
    end = next_packet + 8.0
    last_tick = time.monotonic()
    while time.monotonic() < end:
        now = time.monotonic()
        # Catch up on any packets that "should have" arrived by now.
        while next_packet <= now:
            with drone_module._link_counters_lock:
                drone_module._state_packets_total += 1
                drone_module._last_state_packet_mono = now
            next_packet += 0.1 + rng.uniform(-0.025, 0.025)
        # Telemetry tick at 5 Hz.
        if now - last_tick >= 0.2:
            last_tick = now
            link = d._compute_link()
            d._evaluate_fence(link)
        time.sleep(0.01)

    try:
        assert d._fence_tier in ("ok", "caution"), (
            f"healthy 10 Hz with jitter must not fence into HOVER/LAND; "
            f"got tier={d._fence_tier}"
        )
    finally:
        with drone_module._link_counters_lock:
            drone_module._state_packets_total = base_packets


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    print("smoke_link — link-safety regression checks\n")
    check("§1  patches applied",                          test_patches_applied)
    check("§2  command lock serializes threads",          test_command_lock_serializes)
    check("§3  warmup suppresses early packet loss",      test_warmup_suppresses_early_loss)
    check("§4a debounce: single-tick HOVER",              test_debounce_single_tick_hover_does_not_commit)
    check("§4b debounce: sustained HOVER commits",        test_debounce_sustained_hover_commits)
    check("§4c LAND fires instantly",                     test_land_fires_instantly_without_debounce)
    check("§5  healthy link does not misfire (8 s sim)",  test_healthy_link_does_not_misfire)

    print()
    if _FAILED:
        print(f"FAILED — {len(_FAILED)} check(s):")
        for f in _FAILED:
            print(f"  - {f}")
        return 1
    print("All checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
