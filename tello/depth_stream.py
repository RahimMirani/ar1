"""Live MiDaS depth visualisation stream.

This is the "show the AI's eyeballs" view for the operator console. It
runs a continuous background thread that:

* pulls the latest decoded RGB frame from the drone,
* runs the same MiDaS small model that :mod:`perception` uses for the
  agent's ``check_path_clear`` decisions,
* renders the camera feed with semantic segmentation under a translucent
  Inferno inverse-depth layer, plus the forward-patch rectangle and
  clear/blocked border overlaid,
* encodes that as a JPEG and stores it in a thread-safe buffer.

A FastAPI route in :mod:`main` exposes the buffer as ``/depth.mjpg``,
mirroring the ``/video.mjpg`` shape so the browser can render it with
a plain ``<img>`` tag.

Cost: MiDaS small on CPU is ~150 ms per frame, so we cap at 3 Hz and
pin the worker to a daemon thread. The cv2.dnn.Net is loaded once at
module level inside :mod:`perception` and shared between this stream
and the agent's depth checks — both call ``net.forward`` against the
same loaded net without locking.

The stream is opt-in. Default state is OFF. The operator console has
a "Depth view" toggle that calls :func:`start` / :func:`stop` here.
On disable we drop the latest JPEG so the next subscriber doesn't see
a stale frame from an earlier session.
"""

from __future__ import annotations

import logging
import os
import tarfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

# We intentionally reach into perception for the cached MiDaS path.
# Keeping the inference call here keeps the live render decoupled
# from the agent's PathCheck dataclass (the two layers can evolve
# independently) while reusing the same loaded net.
from perception import _midas_inverse_depth  # noqa: F401

logger = logging.getLogger("tello.depth_stream")


# --------------------------------------------------------------------------- #
# Tuning knobs
# --------------------------------------------------------------------------- #

DEPTH_HZ = 3.0                       # MiDaS ~150 ms / call on CPU; 3 Hz keeps
                                     # one core ~50 % loaded and leaves the
                                     # agent's on-demand check_path_clear
                                     # calls feeling instant.
DEPTH_JPEG_QUALITY = 70              # same encoder quality as /video.mjpg
DEPTH_OUTPUT_W = 480                 # render dimensions; tile is ~480 wide
DEPTH_OUTPUT_H = 360                 # in the 40 % UI tile.
DEPTH_ALPHA = float(os.getenv("FIREDRONE_DEPTH_ALPHA", "0.42"))

# These two thresholds MUST match perception.check_path_clear's defaults
# (near_threshold + obstacle_max_ratio). If you change one, change both,
# or move the logic into a shared helper in perception.py.
PATCH_NEAR_THRESHOLD = 0.78
PATCH_OBSTACLE_MAX_RATIO = 0.40

# Fast local semantic segmentation overlay. We keep this in OpenCV DNN so
# it fits the existing no-Torch runtime and remains portable on the M1 Pro
# MacBook. The model is a small DeepLabV3/MobileNetV2 graph trained on
# PASCAL VOC classes; it is best used as an operator aid ("person/table/chair
# shaped blob here"), not as a safety gate.
SEGMENTATION_ENABLED = os.getenv("FIREDRONE_SEGMENTATION", "1").lower() not in {"0", "false", "off"}
SEGMENTATION_INPUT_SIZE = int(os.getenv("FIREDRONE_SEGMENTATION_SIZE", "257"))
SEGMENTATION_ALPHA = float(os.getenv("FIREDRONE_SEGMENTATION_ALPHA", "0.52"))
SEGMENTATION_MODEL_URL = os.getenv(
    "FIREDRONE_SEGMENTATION_URL",
    "http://download.tensorflow.org/models/"
    "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz",
)
SEGMENTATION_CACHE_DIR = Path.home() / ".cache" / "firedrone" / "models" / "deeplabv3_mnv2"
SEGMENTATION_GRAPH = SEGMENTATION_CACHE_DIR / "frozen_inference_graph.pb"

PASCAL_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train",
    "tv/monitor",
]

PASCAL_COLORS = np.array(
    [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
        (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
        (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
        (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
        (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    ],
    dtype=np.uint8,
)

_seg_net: cv2.dnn.Net | None = None
_seg_lock = threading.Lock()


@dataclass
class DepthStreamStatus:
    enabled: bool
    last_render_ms: int | None
    last_obstacle_ratio: float | None
    last_clear: bool | None
    midas_available: bool | None
    segmentation_available: bool | None
    last_segmentation_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DepthStream:
    """Background MiDaS render -> JPEG buffer + status accessor."""

    def __init__(self, get_frame: Callable[[], Optional[np.ndarray]]) -> None:
        self._get_frame = get_frame
        self._enabled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_lock = threading.Lock()
        self._last_render_ms: int | None = None
        self._last_obstacle_ratio: float | None = None
        self._last_clear: bool | None = None
        self._midas_available: bool | None = None
        self._segmentation_available: bool | None = None
        self._last_segmentation_ms: int | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def status(self) -> DepthStreamStatus:
        return DepthStreamStatus(
            enabled=self._enabled,
            last_render_ms=self._last_render_ms,
            last_obstacle_ratio=self._last_obstacle_ratio,
            last_clear=self._last_clear,
            midas_available=self._midas_available,
            segmentation_available=self._segmentation_available,
            last_segmentation_ms=self._last_segmentation_ms,
        )

    def start(self) -> DepthStreamStatus:
        if self._enabled:
            return self.status()
        self._stop.clear()
        self._enabled = True
        t = threading.Thread(target=self._run, name="tello-depth-stream", daemon=True)
        self._thread = t
        t.start()
        logger.info("depth stream started @ %.1f Hz", DEPTH_HZ)
        return self.status()

    def stop(self) -> DepthStreamStatus:
        if not self._enabled:
            return self.status()
        self._enabled = False
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=1.0)
        with self._latest_lock:
            self._latest_jpeg = None
        logger.info("depth stream stopped")
        return self.status()

    def latest_jpeg(self) -> bytes | None:
        """Return the most recent rendered JPEG, or None if not running yet."""
        with self._latest_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------ #
    # worker
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        period = 1.0 / DEPTH_HZ
        next_deadline = time.monotonic()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DEPTH_JPEG_QUALITY]

        while not self._stop.is_set():
            next_deadline += period
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                self._stop.wait(sleep_for)
            else:
                next_deadline = time.monotonic()
            if self._stop.is_set():
                break

            t0 = time.monotonic()
            frame = self._get_frame()
            if frame is None:
                continue

            try:
                viz = self._render(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("depth render failed: %s", exc)
                continue

            ok, buf = cv2.imencode(".jpg", viz, encode_params)
            if not ok:
                continue
            with self._latest_lock:
                self._latest_jpeg = buf.tobytes()
            self._last_render_ms = int((time.monotonic() - t0) * 1000)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _render(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compose the visualisation.

        Bottom to top: camera frame, semantic segmentation colors, translucent
        inverse-depth heatmap, then safety annotations. Returns a BGR uint8
        image ready for cv2.imencode.
        """

        inv = _midas_inverse_depth(frame_bgr)
        if inv is None:
            self._midas_available = False
            self._last_obstacle_ratio = None
            self._last_clear = None
            return self._render_placeholder(frame_bgr)

        self._midas_available = True

        # Per-frame min-max normalisation, identical to perception.check_path_clear,
        # so the operator sees the exact data the safety layer reasons on.
        inv_min = float(inv.min())
        inv_max = float(inv.max())
        norm = (inv - inv_min) / (inv_max - inv_min + 1e-6)
        norm_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        coloured = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)

        # Central forward patch — same coordinates as check_path_clear's
        # "forward" branch (h/4..3h/4, w/4..3w/4 on the inv_depth grid).
        h, w = norm.shape
        patch = norm[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        obstacle_mask = patch > PATCH_NEAR_THRESHOLD
        obstacle_ratio = float(obstacle_mask.mean())
        clear = obstacle_ratio <= PATCH_OBSTACLE_MAX_RATIO
        self._last_obstacle_ratio = obstacle_ratio
        self._last_clear = clear

        # Up-sample to UI dimensions and composite. Depth is deliberately
        # translucent so the segmentation overlay stays visible.
        base = cv2.resize(frame_bgr, (DEPTH_OUTPUT_W, DEPTH_OUTPUT_H))
        seg_t0 = time.monotonic()
        seg_overlay = _segmentation_overlay(frame_bgr, (DEPTH_OUTPUT_W, DEPTH_OUTPUT_H))
        self._last_segmentation_ms = int((time.monotonic() - seg_t0) * 1000)
        if seg_overlay is not None:
            self._segmentation_available = True
            viz = cv2.addWeighted(base, 1.0 - SEGMENTATION_ALPHA, seg_overlay, SEGMENTATION_ALPHA, 0)
        else:
            self._segmentation_available = False
            viz = base
        depth_layer = cv2.resize(coloured, (DEPTH_OUTPUT_W, DEPTH_OUTPUT_H))
        viz = cv2.addWeighted(viz, 1.0 - DEPTH_ALPHA, depth_layer, DEPTH_ALPHA, 0)
        h2, w2 = viz.shape[:2]

        # Patch rectangle in viz coordinates.
        x1, y1 = w2 // 4, h2 // 4
        x2, y2 = 3 * w2 // 4, 3 * h2 // 4
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Coloured border: green when clear, red when blocked. BGR.
        border = (60, 200, 80) if clear else (60, 60, 235)
        cv2.rectangle(viz, (0, 0), (w2 - 1, h2 - 1), border, 4)

        # Top-left stats. Use a black drop-shadow under the white text so
        # it stays legible against bright inferno regions.
        status_text = "CLEAR" if clear else "BLOCKED"
        _draw_text(viz, f"forward: {status_text}", (12, 24), 0.6, 2)
        _draw_text(viz, f"obstacle ratio: {obstacle_ratio * 100:.0f}%",
                   (12, 46), 0.5, 1)
        if self._segmentation_available:
            _draw_text(viz, f"seg {self._last_segmentation_ms} ms",
                       (12, 68), 0.45, 1, faint=True)
        if self._last_render_ms is not None:
            _draw_text(viz, f"MiDaS {self._last_render_ms} ms",
                       (12, h2 - 14), 0.45, 1, faint=True)

        # Tiny legend bottom-right: "near" / "far" with colour swatches.
        _draw_legend(viz)
        _draw_segmentation_legend(viz, seg_overlay)

        return viz

    def _render_placeholder(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Fallback view when MiDaS hasn't loaded yet."""
        small = cv2.resize(frame_bgr, (DEPTH_OUTPUT_W, DEPTH_OUTPUT_H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(viz, (0, 0), (viz.shape[1] - 1, viz.shape[0] - 1),
                      (60, 180, 220), 4)
        _draw_text(viz, "MiDaS unavailable", (12, 28), 0.7, 2)
        _draw_text(viz, "depth view will resume once the model loads",
                   (12, 50), 0.45, 1, faint=True)
        return viz


# --------------------------------------------------------------------------- #
# segmentation helpers
# --------------------------------------------------------------------------- #


def _download_segmentation_model() -> Path:
    if SEGMENTATION_GRAPH.exists() and SEGMENTATION_GRAPH.stat().st_size > 1_000_000:
        return SEGMENTATION_GRAPH

    SEGMENTATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = SEGMENTATION_CACHE_DIR / "deeplabv3_mnv2_pascal.tar.gz"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        logger.info("downloading segmentation model from %s", SEGMENTATION_MODEL_URL)
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(SEGMENTATION_MODEL_URL, timeout=90) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                dst.write(chunk)
        tmp.replace(archive)

    with tarfile.open(archive, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("frozen_inference_graph.pb")),
            None,
        )
        if member is None:
            raise RuntimeError("segmentation archive did not contain frozen_inference_graph.pb")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError("could not extract segmentation graph")
        with open(SEGMENTATION_GRAPH, "wb") as dst:
            while True:
                chunk = extracted.read(1 << 16)
                if not chunk:
                    break
                dst.write(chunk)

    return SEGMENTATION_GRAPH


def _load_segmentation_net() -> cv2.dnn.Net | None:
    if not SEGMENTATION_ENABLED:
        return None

    global _seg_net
    with _seg_lock:
        if _seg_net is not None:
            return _seg_net
        try:
            graph = _download_segmentation_model()
            net = cv2.dnn.readNetFromTensorflow(str(graph))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            _seg_net = net
            logger.info(
                "segmentation model loaded (%s, input=%d)",
                graph,
                SEGMENTATION_INPUT_SIZE,
            )
            return net
        except Exception as exc:  # noqa: BLE001
            logger.warning("segmentation model unavailable: %s", exc)
            return None


def _segmentation_overlay(frame_bgr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray | None:
    net = _load_segmentation_net()
    if net is None:
        return None

    inp = cv2.resize(frame_bgr, (SEGMENTATION_INPUT_SIZE, SEGMENTATION_INPUT_SIZE))
    blob = cv2.dnn.blobFromImage(inp, 1.0, (SEGMENTATION_INPUT_SIZE, SEGMENTATION_INPUT_SIZE),
                                 (0, 0, 0), swapRB=True, crop=False)
    try:
        net.setInput(blob)
        out = net.forward()
    except Exception as exc:  # noqa: BLE001
        logger.debug("segmentation inference failed: %s", exc)
        return None

    class_map = _class_map_from_output(out)
    if class_map is None:
        return None

    class_map = cv2.resize(class_map, out_size, interpolation=cv2.INTER_NEAREST)
    colours = PASCAL_COLORS[class_map % len(PASCAL_COLORS)]
    # Keep background transparent-by-convention: the caller blends this overlay
    # over the camera frame, so black background contributes little.
    colours[class_map == 0] = (0, 0, 0)

    edges = cv2.Canny(class_map, 0, 1)
    colours[edges > 0] = (255, 255, 255)
    return colours


def _class_map_from_output(out: np.ndarray) -> np.ndarray | None:
    """Handle both DeepLab output styles: class IDs or class logits."""
    if out.ndim == 4:
        if out.shape[1] == 1:
            return out[0, 0].astype(np.uint8)
        if out.shape[-1] == 1:
            return out[0, :, :, 0].astype(np.uint8)
        if out.shape[1] <= 32:
            return out[0].argmax(axis=0).astype(np.uint8)
        if out.shape[-1] <= 32:
            return out[0].argmax(axis=-1).astype(np.uint8)
    if out.ndim == 3:
        if out.shape[0] == 1:
            return out[0].astype(np.uint8)
        if out.shape[0] <= 32:
            return out.argmax(axis=0).astype(np.uint8)
        if out.shape[-1] <= 32:
            return out.argmax(axis=-1).astype(np.uint8)
    if out.ndim == 2:
        return out.astype(np.uint8)
    return None


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    thickness: int,
    *,
    faint: bool = False,
) -> None:
    """White text with a single-pixel black drop-shadow for legibility
    against the inferno colourmap."""
    cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    fg = (210, 210, 210) if faint else (255, 255, 255)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg,
                thickness, cv2.LINE_AA)


def _draw_legend(img: np.ndarray) -> None:
    """Tiny near/far legend in the bottom-right corner."""
    h, w = img.shape[:2]
    bar_w, bar_h = 90, 8
    x0, y0 = w - bar_w - 14, h - bar_h - 28
    # Build a horizontal inferno gradient strip.
    grad = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :]
    grad = np.repeat(grad, bar_h, axis=0)
    strip = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)
    img[y0:y0 + bar_h, x0:x0 + bar_w] = strip
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h),
                  (255, 255, 255), 1)
    _draw_text(img, "far", (x0 - 22, y0 + bar_h - 0), 0.4, 1, faint=True)
    _draw_text(img, "near", (x0 + bar_w + 4, y0 + bar_h - 0), 0.4, 1, faint=True)


def _draw_segmentation_legend(img: np.ndarray, seg_overlay: np.ndarray | None) -> None:
    if seg_overlay is None:
        return
    h, _w = img.shape[:2]
    y = h - 42
    x = 14
    _draw_text(img, "seg + translucent depth", (x, y), 0.42, 1, faint=True)
