"""Local perception layer — depth check + optical-flow watchdog.

The Tello has no physical depth sensor (just the downward ToF for
altitude). For obstacle avoidance we run two complementary signals
on-laptop, both cheap enough for a Tello-class platform:

* **MiDaS v2.1 Small** (ONNX, ~21 MB) — monocular inverse depth. Called
  on demand by :func:`check_path_clear` before the agent commits to a
  forward move. ~150 ms per call on CPU.
* **Farnebäck dense optical flow** — runs as a continuous background
  watchdog at ~10 Hz, evaluates radial divergence in the central patch
  of the frame. A spike means something is rushing at the camera; we
  publish a ``perception_alert`` event and call ``drone.set_velocity(0)``
  to hover. This is the reactive safety net while the agent's deliberate
  depth checks are the proactive plan.

MiDaS file is downloaded once into ``~/.cache/firedrone/models/`` and
mmapped from there on subsequent runs. If download fails the depth
check degrades gracefully (returns ``{"available": False}``) — optical
flow still works.

Note on the Tello safety contract: we never call any djitellopy method
directly from this module. The only drone interaction is through the
shared ``Drone`` instance the caller passes in, and we use its existing
``set_velocity`` method (the documented fire-and-forget RC path).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from events import bus

logger = logging.getLogger("tello.perception")

# --------------------------------------------------------------------------- #
# MiDaS depth
# --------------------------------------------------------------------------- #

MIDAS_MODEL_URL = os.getenv(
    "FIREDRONE_MIDAS_URL",
    # Official MiDaS v2.1 small ONNX release asset.
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
)
MIDAS_INPUT_SIZE = 256
MIDAS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MIDAS_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIDAS_CACHE = Path.home() / ".cache" / "firedrone" / "models" / "midas_small.onnx"

_midas_net: cv2.dnn.Net | None = None
_midas_lock = threading.Lock()


def _download_midas() -> Path:
    """Fetch the MiDaS Small ONNX into the local cache, return the path."""
    if MIDAS_CACHE.exists() and MIDAS_CACHE.stat().st_size > 5_000_000:
        return MIDAS_CACHE
    MIDAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading MiDaS model from %s", MIDAS_MODEL_URL)
    tmp = MIDAS_CACHE.with_suffix(".part")
    with urllib.request.urlopen(MIDAS_MODEL_URL, timeout=60) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 16)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(MIDAS_CACHE)
    logger.info("MiDaS model cached at %s (%.1f MB)", MIDAS_CACHE, MIDAS_CACHE.stat().st_size / 1e6)
    return MIDAS_CACHE


def _load_midas() -> cv2.dnn.Net | None:
    """Lazy-load the MiDaS net. Returns None on any failure."""
    global _midas_net
    with _midas_lock:
        if _midas_net is not None:
            return _midas_net
        try:
            path = _download_midas()
            net = cv2.dnn.readNet(str(path))
            # CPU only — we don't assume a CUDA OpenCV build.
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            _midas_net = net
            return net
        except Exception as exc:
            logger.warning("MiDaS load failed: %s", exc)
            return None


def _midas_inverse_depth(bgr_frame: np.ndarray) -> np.ndarray | None:
    """Return the MiDaS inverse-depth map at MIDAS_INPUT_SIZE resolution."""
    net = _load_midas()
    if net is None:
        return None
    img = cv2.resize(bgr_frame, (MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MIDAS_MEAN) / MIDAS_STD
    blob = img.transpose(2, 0, 1)[None]
    net.setInput(blob)
    return net.forward()[0]


@dataclass
class PathCheck:
    """Result of a single :func:`check_path_clear` call."""

    available: bool
    clear: bool
    direction: str
    min_depth_norm: float  # 0=far, 1=near (normalised within frame)
    obstacle_ratio: float  # fraction of patch flagged as near
    center_closeness: float  # 0=far, 1=near for the central "what's ahead" box
    estimated_center_distance_m: float | None  # coarse non-metric estimate
    distance_band: str  # "far" | "mid" | "near" | "very_near" | "unknown"
    latency_ms: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_path_clear(
    frame_bgr: np.ndarray,
    *,
    direction: str = "forward",
    near_threshold: float = 0.78,
    obstacle_max_ratio: float = 0.40,
) -> PathCheck:
    """Decide if a flight direction is safe given a single camera frame.

    MiDaS produces *inverse* depth — large values mean close. We grab the
    central forward patch (for ``direction == "forward"``) or the left/
    right halves, normalise across the frame, and flag the share of
    pixels whose normalised value is above ``near_threshold``. If that
    share exceeds ``obstacle_max_ratio`` we declare the path blocked.

    Threshold tuning note: because we normalise *per frame*, the
    nearest pixel in any view always scores 1.0, even if it's actually
    several metres out. With a global ``near_threshold`` of 0.55 and
    ``obstacle_max_ratio`` of 0.20 (early defaults) realistic indoor
    scenes — where the central patch naturally contains "in front of
    me" content — got refused almost universally. The current defaults
    (0.78 / 0.40) reject the cases the watchdog wouldn't already catch
    (a wall or person filling most of the camera) without false-firing
    on a normal cluttered living room. Both remain tunable by the
    caller.
    """
    t0 = time.monotonic()
    inv_depth = _midas_inverse_depth(frame_bgr)
    if inv_depth is None:
        return PathCheck(
            available=False,
            clear=True,
            direction=direction,
            min_depth_norm=0.0,
            obstacle_ratio=0.0,
            center_closeness=0.0,
            estimated_center_distance_m=None,
            distance_band="unknown",
            latency_ms=int((time.monotonic() - t0) * 1000),
            reason="MiDaS unavailable — depth check skipped",
        )

    h, w = inv_depth.shape
    norm = (inv_depth - inv_depth.min()) / (np.ptp(inv_depth) + 1e-6)

    if direction == "forward":
        patch = norm[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    elif direction == "left":
        patch = norm[h // 4 : 3 * h // 4, : w // 2]
    elif direction == "right":
        patch = norm[h // 4 : 3 * h // 4, w // 2 :]
    elif direction == "up":
        patch = norm[: h // 3, w // 4 : 3 * w // 4]
    elif direction == "down":
        patch = norm[2 * h // 3 :, w // 4 : 3 * w // 4]
    else:
        patch = norm[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    obstacle_mask = patch > near_threshold
    obstacle_ratio = float(obstacle_mask.mean())
    min_depth_norm = float(patch.max())
    center_patch = norm[3 * h // 8 : 5 * h // 8, 3 * w // 8 : 5 * w // 8]
    center_closeness = float(np.percentile(center_patch, 90)) if center_patch.size else min_depth_norm
    estimated_distance_m = _estimate_center_distance_m(center_closeness)
    distance_band = _distance_band(center_closeness)
    blocked = (
        obstacle_ratio > obstacle_max_ratio
        or center_closeness >= 0.92
        or (estimated_distance_m is not None and estimated_distance_m < 0.55)
    )

    reason = (
        f"clear ({obstacle_ratio * 100:.0f}% near, center ~{estimated_distance_m:.1f} m)"
        if not blocked
        else (
            f"BLOCKED — {obstacle_ratio * 100:.0f}% near, "
            f"center object ~{estimated_distance_m:.1f} m ({distance_band})"
        )
    )

    return PathCheck(
        available=True,
        clear=not blocked,
        direction=direction,
        min_depth_norm=min_depth_norm,
        obstacle_ratio=obstacle_ratio,
        center_closeness=center_closeness,
        estimated_center_distance_m=estimated_distance_m,
        distance_band=distance_band,
        latency_ms=int((time.monotonic() - t0) * 1000),
        reason=reason,
    )


def _estimate_center_distance_m(closeness: float) -> float:
    """Coarse monocular distance estimate for the center of frame.

    MiDaS Small is relative inverse depth, not a range sensor. This maps the
    normalized center closeness into conservative indoor buckets so the agent
    can reason "too close / enough room" without pretending centimetre accuracy.
    """
    c = max(0.0, min(1.0, float(closeness)))
    # Piecewise inverse-ish curve: 0.25 -> ~2.6 m, 0.55 -> ~1.3 m,
    # 0.80 -> ~0.7 m, 0.95 -> ~0.35 m.
    return round(max(0.30, min(3.0, 3.0 - 2.9 * (c ** 1.7))), 2)


def _distance_band(closeness: float) -> str:
    c = max(0.0, min(1.0, float(closeness)))
    if c >= 0.88:
        return "very_near"
    if c >= 0.70:
        return "near"
    if c >= 0.45:
        return "mid"
    return "far"


# --------------------------------------------------------------------------- #
# Optical-flow watchdog
# --------------------------------------------------------------------------- #

WATCHDOG_HZ = 10.0
FLOW_SPIKE_THRESHOLD = 1.2  # mean radial divergence in pixels / frame
# Hold-off after firing so we don't spam alerts during a stop.
FLOW_COOLDOWN_SEC = 1.5


@dataclass
class WatchdogStatus:
    enabled: bool
    flow_score: float | None
    last_alert_ts: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpticalFlowWatchdog:
    """Background thread that hovers the drone when the central optical-flow
    expansion exceeds a threshold.

    The watchdog never *moves* the drone — it only sends an
    ``set_velocity(0, 0, 0, 0)`` to halt any in-progress motion, and
    publishes a ``perception_alert`` event so the operator and the
    dispatcher can see the trigger. The agent's reasoning loop sees the
    same event and can decide whether to back off or continue.
    """

    def __init__(self, get_frame: Callable[[], np.ndarray | None], drone_stop: Callable[[], None]) -> None:
        self._get_frame = get_frame
        self._drone_stop = drone_stop
        self._enabled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_score: float | None = None
        self._last_alert: float | None = None

    def status(self) -> WatchdogStatus:
        return WatchdogStatus(
            enabled=self._enabled,
            flow_score=self._last_score,
            last_alert_ts=self._last_alert,
        )

    def start(self) -> WatchdogStatus:
        if self._enabled:
            return self.status()
        self._stop.clear()
        self._enabled = True
        t = threading.Thread(target=self._run, name="tello-perception", daemon=True)
        self._thread = t
        t.start()
        bus.publish_threadsafe(
            {"type": "perception_state", "enabled": True, "source": "watchdog"}
        )
        return self.status()

    def stop(self) -> WatchdogStatus:
        if not self._enabled:
            return self.status()
        self._enabled = False
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=1.0)
        bus.publish_threadsafe(
            {"type": "perception_state", "enabled": False, "source": "watchdog"}
        )
        return self.status()

    def _run(self) -> None:
        period = 1.0 / WATCHDOG_HZ
        prev_gray: np.ndarray | None = None
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            next_deadline += period
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                self._stop.wait(sleep_for)
            else:
                next_deadline = time.monotonic()

            frame = self._get_frame()
            if frame is None:
                prev_gray = None
                continue

            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None or prev_gray.shape != gray.shape:
                prev_gray = gray
                continue

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 21, 2, 5, 1.1, 0,
                )
            except Exception as exc:
                logger.warning("optical flow failed: %s", exc)
                prev_gray = gray
                continue
            prev_gray = gray

            score = _flow_expansion(flow)
            self._last_score = score

            now = time.time()
            cooled = (self._last_alert is None) or (now - self._last_alert > FLOW_COOLDOWN_SEC)
            if score > FLOW_SPIKE_THRESHOLD and cooled:
                self._last_alert = now
                logger.info("flow spike score=%.2f -> hover", score)
                try:
                    self._drone_stop()
                except Exception as exc:
                    logger.warning("flow-watchdog drone_stop failed: %s", exc)
                bus.publish_threadsafe(
                    {
                        "type": "perception_alert",
                        "kind": "flow_spike",
                        "score": score,
                        "threshold": FLOW_SPIKE_THRESHOLD,
                        "action": "hover",
                        "reason": (
                            f"optical-flow expansion {score:.2f} exceeded "
                            f"{FLOW_SPIKE_THRESHOLD:.2f} — likely incoming obstacle"
                        ),
                    }
                )


def _flow_expansion(flow: np.ndarray) -> float:
    """Mean outward (radial) component of a dense flow field.

    A positive number means the central scene is *expanding* in the
    image — characteristic of the camera approaching it.
    """
    h, w = flow.shape[:2]
    cy, cx = h // 2, w // 2
    half = min(h, w) // 4
    yy, xx = np.mgrid[-half:half, -half:half].astype(np.float32)
    norm = np.sqrt(xx ** 2 + yy ** 2) + 1e-6
    ux, uy = xx / norm, yy / norm
    patch = flow[cy - half : cy + half, cx - half : cx + half]
    radial = patch[..., 0] * ux + patch[..., 1] * uy
    return float(np.mean(radial))
