"""2D pose + occupancy mapping (Tier 1: dead-reckoning + MiDaS).

This is the "where am I and what's around me" layer that sits next to
:mod:`perception` and :mod:`depth_stream`. Like both of those it is a
pure consumer — it never touches ``djitellopy`` directly. Inputs come
in via the ``Drone`` accessors the caller plumbs through:

* ``get_telemetry()`` → the cached dict published by ``drone.snapshot().telemetry``
* ``get_frame()``     → the latest BGR camera frame (or ``None``)
* ``get_flying()``    → bool from ``drone.snapshot().flying``

What it produces:

* a 2D world-frame **pose** ``(x_m, y_m, theta_rad)`` anchored at
  takeoff (0,0 → +x = the direction the camera was looking at takeoff,
  +y = the drone's right);
* a rolling **trajectory** of past poses (capped for memory);
* a coarse **occupancy grid** built from MiDaS depth observations stamped
  into world coordinates;
* a JPEG **render** of all three for the operator/dispatcher UI.

The integration is honest dead-reckoning — no loop closure. Realistic
drift on a textured indoor floor is 30-60 cm/min with the IMU-aware
smoothing below, and unbounded once the belly optical flow loses lock.
That's good enough for "the drone made a floor plan of the room while
deciding" but not for waypoint navigation across rooms.

Why a separate module:

* Owning pose state means we get to choose the integration rate (10 Hz),
  smoothing strategy, and reset semantics independently of telemetry.
* Keeping the MiDaS-based obstacle stamping here (not in ``perception``)
  lets ``check_path_clear`` stay a pure per-call function. Mapping
  accumulates; safety checks don't.
* It respects the AGENTS.md contract: ``drone.py`` remains the only
  module that talks to ``djitellopy``.

IMU usage — what we get out of agx/agy/agz
------------------------------------------

The Tello firmware already fuses IMU + belly-cam optical flow internally
to produce ``speed_x/y/z``, so we're not doing VIO. What we *do* get
from the raw accel reads (in milli-g, body-frame) on top of that:

1. **Velocity smoothing between telemetry ticks.** Telemetry samples at
   ~10 Hz; raw accel is at the same rate but tracks the *derivative* of
   velocity. A complementary filter blend lowers the noise on the
   position integral.
2. **Belly-cam lockout detection (the load-bearing one).** When the
   floor is too featureless / dark / shiny, the Tello's optical-flow
   sensor stops contributing and ``speed_x/y`` reads near zero. Without
   the IMU you can't tell that from a hover. With the IMU, if accel is
   non-trivial while velocity is ~0 over several ticks, the integral is
   lying. We tag those poses as low-confidence and stop stamping
   obstacles during the affected window.
3. **Pitch/roll de-tilt of camera bearings.** When projecting "blocked
   forward" from a MiDaS reading we already know the body heading from
   yaw; pitch/roll on top tell us the actual camera bearing in 3D so
   the obstacle lands at the right map angle.
4. **Zero-velocity hold while hovering.** When |accel - g| and |vel|
   are both near zero across several ticks, the drone is in steady
   hover and we should freeze the pose instead of integrating noise.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

# Reuse the cached MiDaS net from perception so we don't load it twice.
from perception import _midas_inverse_depth  # noqa: F401

logger = logging.getLogger("tello.mapping")


# --------------------------------------------------------------------------- #
# Tuning knobs — grouped so the trade-offs are visible in one place.
#
# These were picked to give the demo a usable in-room view without
# pretending the system has metric accuracy. Comments explain *why* each
# value, not what it does.
# --------------------------------------------------------------------------- #

# Pose-integration loop rate. Matches the Tello state broadcast rate of
# ~10 Hz; integrating faster than the data refreshes just adds noise.
POSE_HZ = 10.0

# Render + obstacle-update loop. MiDaS is ~150 ms on CPU so 2 Hz is
# comfortable; the JPEG itself takes a couple of ms.
RENDER_HZ = 2.0

# Grid geometry. 10 m square, 5 cm cells -> 200×200 array. Living-room
# scale. The grid is centred on the takeoff origin: world point (x, y)
# in metres maps to cell index (cy + y/cell, cx + x/cell). That keeps
# the math simple at the cost of refusing to map flights that wander
# outside the box, which is fine for the fire-response demo.
CELL_SIZE_M = 0.05
GRID_EXTENT_M = 10.0
GRID_CELLS = int(GRID_EXTENT_M / CELL_SIZE_M)  # 200

# Log-odds clamps for the occupancy grid. Standard SLAM-textbook values;
# a pixel that gets "near" stamped repeatedly saturates at +LOG_ODDS_MAX,
# a pixel that gets "free" stamped repeatedly saturates at -LOG_ODDS_MAX.
LOG_ODDS_OCC_HIT  = 0.7
LOG_ODDS_FREE_HIT = 0.35
LOG_ODDS_MAX      = 4.5
LOG_ODDS_MIN      = -4.5
LOG_ODDS_OCC_THRESHOLD = 0.8   # render threshold to call a cell "wall"
LOG_ODDS_FREE_THRESHOLD = -0.4 # render threshold to call a cell "open"

# Tello camera horizontal field of view. The original Tello front camera
# is ~82° diagonal; the H FOV is roughly 67°. We use this to map MiDaS
# columns to world bearings.
CAMERA_H_FOV_DEG = 67.0

# MiDaS thresholds matching ``perception.check_path_clear`` defaults so
# the two layers agree on what "blocked" means.
NEAR_THRESHOLD = 0.78
OBSTACLE_MAX_RATIO = 0.40

# Crude metric distance buckets for MiDaS-near observations. MiDaS is
# non-metric (inverse depth, normalised per frame), so any single number
# would be a lie. We stamp obstacles at a fan from MIN_M to MAX_M in
# the cell at the bearing, which gives a usable visual of "there is
# *something* in that direction within arm's reach" without overclaiming
# precise range.
OBS_STAMP_NEAR_M = 0.8
OBS_STAMP_FAR_M  = 2.2

# Free-space stamping range — when the path is clear forward, we mark
# every cell in the fan up to FREE_M as "seen as free". Stops shorter
# than the obstacle stamp because past 3 m a monocular view doesn't
# really tell you "this is open".
FREE_STAMP_M = 3.0

# Belly-cam lockout detection. The IMU acceleration is in milli-g, so
# 80 mg ≈ 0.8 m/s². Translated: if the drone reads <3 cm/s of body
# velocity for >0.6 s while horizontal IMU accel is >80 mg, the flow
# sensor has probably dropped lock and the position integral is lying.
LOCKOUT_VEL_CMPS = 3.0
LOCKOUT_ACCEL_MG = 80.0
LOCKOUT_CONSECUTIVE_TICKS = 6  # at 10 Hz → 0.6 s

# Complementary-filter weight for velocity smoothing. ``alpha`` blends
# the IMU-projected previous velocity (vel_prev + accel * dt) against
# the latest measurement. 0.3 favours the measurement but lets accel
# fill in the high-frequency component between ticks.
VEL_SMOOTH_ALPHA = 0.3

# Trajectory cap. 6000 entries at 10 Hz = 10 minutes; beyond that we
# drop the oldest. Past trajectory is rendered as a thin polyline so
# this is purely a memory bound.
TRAJECTORY_MAX = 6000

# Render dimensions. Same shape as the depth tile so the side-by-side
# layout in the operator console feels consistent.
RENDER_W = 480
RENDER_H = 480
JPEG_QUALITY = 75


@dataclass
class MapStatus:
    enabled: bool
    pose: dict[str, float]
    pose_confidence: str          # "ok" | "low"
    flying: bool
    trajectory_points: int
    occupied_cells: int
    free_cells: int
    started_at: float | None
    last_pose_ms: int | None
    last_render_ms: int | None
    midas_available: bool | None
    lockout_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MapSnapshot:
    """Lightweight read of pose + trajectory + obstacle summary.

    The agent's ``get_pose`` and ``get_map_summary`` tools read this; it
    deliberately excludes the bulky occupancy grid so the LLM context
    doesn't bloat. The renderer accesses the full grid directly.
    """

    pose: dict[str, float]
    pose_confidence: str
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    occupied_cells: int = 0
    free_cells: int = 0
    bounds_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Mapper:
    """Background pose integrator + occupancy renderer.

    Threading model mirrors :class:`perception.OpticalFlowWatchdog` and
    :class:`depth_stream.DepthStream`:

    * ``_pose_loop``   — 10 Hz, integrates yaw + speeds into world-frame
      pose, watches IMU vs flow disagreement for lockout, appends to
      the trajectory ring buffer.
    * ``_render_loop`` —  2 Hz, pulls one camera frame, runs MiDaS once,
      stamps observed bearings into the log-odds grid, renders the JPEG.

    Both threads are daemons. ``start()`` is idempotent; ``stop()`` joins
    with a short timeout so the FastAPI shutdown path doesn't block.
    """

    def __init__(
        self,
        get_telemetry: Callable[[], dict[str, Any]],
        get_frame:     Callable[[], Optional[np.ndarray]],
        get_flying:    Callable[[], bool],
    ) -> None:
        self._get_tele   = get_telemetry
        self._get_frame  = get_frame
        self._get_flying = get_flying

        # Lifecycle.
        self._enabled = False
        self._stop = threading.Event()
        self._pose_thread:   threading.Thread | None = None
        self._render_thread: threading.Thread | None = None

        # Pose + trajectory.
        self._state_lock = threading.Lock()
        self._pose_x_m: float = 0.0
        self._pose_y_m: float = 0.0
        self._pose_theta_rad: float = 0.0
        self._pose_confidence: str = "ok"
        self._lockout_count: int = 0
        self._lockout_active: bool = False
        self._vel_smooth_body: tuple[float, float] = (0.0, 0.0)  # cm/s in body
        self._trajectory: deque[tuple[float, float, str]] = deque(maxlen=TRAJECTORY_MAX)
        self._was_flying: bool = False
        self._started_at: float | None = None
        self._last_pose_mono: float | None = None
        self._last_pose_ms: int | None = None

        # Occupancy grid (log-odds). Pre-allocated zeros; the centre cell
        # (GRID_CELLS // 2, GRID_CELLS // 2) is the takeoff origin.
        self._grid_lock = threading.Lock()
        self._grid: np.ndarray = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)

        # Render buffer.
        self._latest_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_render_ms: int | None = None
        self._midas_available: bool | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def start(self) -> MapStatus:
        if self._enabled:
            return self.status()
        self._stop.clear()
        self._enabled = True
        self._pose_thread = threading.Thread(
            target=self._pose_loop, name="tello-mapper-pose", daemon=True
        )
        self._render_thread = threading.Thread(
            target=self._render_loop, name="tello-mapper-render", daemon=True
        )
        self._pose_thread.start()
        self._render_thread.start()
        logger.info("mapper started — pose %.0f Hz, render %.0f Hz", POSE_HZ, RENDER_HZ)
        return self.status()

    def stop(self) -> MapStatus:
        if not self._enabled:
            return self.status()
        self._enabled = False
        self._stop.set()
        for t in (self._pose_thread, self._render_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._pose_thread = self._render_thread = None
        with self._latest_lock:
            self._latest_jpeg = None
        logger.info("mapper stopped")
        return self.status()

    def reset(self) -> MapStatus:
        """Zero the pose, drop the trajectory, clear the grid.

        Called automatically when we observe a False→True transition in
        ``flying`` (so each takeoff starts on a fresh canvas), and
        exposed manually so the operator can re-anchor mid-flight.
        """
        with self._state_lock:
            self._pose_x_m = 0.0
            self._pose_y_m = 0.0
            self._pose_theta_rad = 0.0
            self._pose_confidence = "ok"
            self._lockout_count = 0
            self._lockout_active = False
            self._vel_smooth_body = (0.0, 0.0)
            self._trajectory.clear()
            self._started_at = time.time()
            self._last_pose_mono = None
        with self._grid_lock:
            self._grid.fill(0.0)
        logger.info("mapper reset — pose=(0,0,0), grid cleared")
        return self.status()

    def status(self) -> MapStatus:
        with self._state_lock:
            pose = self._pose_dict()
            confidence = self._pose_confidence
            n_traj = len(self._trajectory)
            started_at = self._started_at
            last_pose_ms = self._last_pose_ms
            lockout = self._lockout_active
        with self._grid_lock:
            occ_n  = int((self._grid > LOG_ODDS_OCC_THRESHOLD).sum())
            free_n = int((self._grid < LOG_ODDS_FREE_THRESHOLD).sum())
        return MapStatus(
            enabled=self._enabled,
            pose=pose,
            pose_confidence=confidence,
            flying=self._was_flying,
            trajectory_points=n_traj,
            occupied_cells=occ_n,
            free_cells=free_n,
            started_at=started_at,
            last_pose_ms=last_pose_ms,
            last_render_ms=self._last_render_ms,
            midas_available=self._midas_available,
            lockout_active=lockout,
        )

    def snapshot(self) -> MapSnapshot:
        """Lightweight read for the agent tools.

        Returns the current pose, the trajectory polyline as `[(x, y),
        ...]` in metres, and aggregate obstacle counts. The trajectory
        is sub-sampled to ≤ 200 points so an LLM tool call doesn't get
        a 6000-element list shoved at it.
        """
        with self._state_lock:
            pose = self._pose_dict()
            confidence = self._pose_confidence
            traj_full = [(x, y) for (x, y, _conf) in self._trajectory]
        if len(traj_full) > 200:
            stride = max(1, len(traj_full) // 200)
            traj = traj_full[::stride]
        else:
            traj = list(traj_full)
        with self._grid_lock:
            occ_n  = int((self._grid > LOG_ODDS_OCC_THRESHOLD).sum())
            free_n = int((self._grid < LOG_ODDS_FREE_THRESHOLD).sum())
            ys, xs = np.where(self._grid > LOG_ODDS_OCC_THRESHOLD)
            if xs.size:
                xs_m = (xs - GRID_CELLS // 2) * CELL_SIZE_M
                ys_m = (ys - GRID_CELLS // 2) * CELL_SIZE_M
                bounds = (
                    float(xs_m.min()), float(ys_m.min()),
                    float(xs_m.max()), float(ys_m.max()),
                )
            else:
                bounds = (0.0, 0.0, 0.0, 0.0)
        return MapSnapshot(
            pose=pose,
            pose_confidence=confidence,
            trajectory=traj,
            occupied_cells=occ_n,
            free_cells=free_n,
            bounds_m=bounds,
        )

    def latest_jpeg(self) -> bytes | None:
        with self._latest_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------ #
    # pose loop
    # ------------------------------------------------------------------ #

    def _pose_loop(self) -> None:
        period = 1.0 / POSE_HZ
        next_deadline = time.monotonic()
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
            try:
                tele = self._get_tele() or {}
            except Exception as exc:  # noqa: BLE001
                logger.debug("mapper telemetry read failed: %s", exc)
                continue
            try:
                flying = bool(self._get_flying())
            except Exception:
                flying = False

            # Auto-reset on takeoff. We watch False→True transitions
            # rather than coupling into drone.takeoff() so the mapper
            # remains a pure consumer (AGENTS.md contract).
            if flying and not self._was_flying:
                logger.info("mapper: takeoff detected — resetting pose + grid")
                self.reset()
            self._was_flying = flying

            if not flying:
                # On the ground: don't integrate, but keep the loop
                # running so we resume cleanly on the next takeoff.
                with self._state_lock:
                    self._last_pose_mono = None
                continue

            self._integrate_pose(tele)
            self._last_pose_ms = int((time.monotonic() - t0) * 1000)

    def _integrate_pose(self, tele: dict[str, Any]) -> None:
        """One pose-loop tick: apply IMU smoothing, detect lockout, update
        ``(x, y, theta)``, append to trajectory.

        Conventions documented at module top: yaw_deg is the world heading
        (0° at takeoff), speed_x is body-forward, speed_y is body-right,
        both in cm/s. We integrate in metres.
        """
        now = time.monotonic()
        with self._state_lock:
            last = self._last_pose_mono
            self._last_pose_mono = now
        if last is None:
            return  # first tick of the flight — no dt yet
        dt = max(0.0, min(0.5, now - last))  # cap dt so a thread stall
        # doesn't teleport the drone halfway across the map

        # --- body-frame velocity (cm/s) with IMU complementary filter ---
        vx_b_meas = _as_float(tele.get("speed_x"), 0.0)
        vy_b_meas = _as_float(tele.get("speed_y"), 0.0)
        # IMU accel in milli-g → cm/s² (1 g ≈ 981 cm/s²; mg × 0.981)
        ax_mg = _as_float(tele.get("accel_x_mg"), 0.0)
        ay_mg = _as_float(tele.get("accel_y_mg"), 0.0)
        ax_cmps2 = ax_mg * 0.981
        ay_cmps2 = ay_mg * 0.981

        vx_prev, vy_prev = self._vel_smooth_body
        # Project previous velocity forward using accel; blend with the
        # latest measurement. alpha favours the measurement (we don't
        # trust the IMU integral to stay calibrated over long windows)
        # but uses accel for the inter-tick high-frequency component.
        vx_pred = vx_prev + ax_cmps2 * dt
        vy_pred = vy_prev + ay_cmps2 * dt
        vx_b = VEL_SMOOTH_ALPHA * vx_pred + (1.0 - VEL_SMOOTH_ALPHA) * vx_b_meas
        vy_b = VEL_SMOOTH_ALPHA * vy_pred + (1.0 - VEL_SMOOTH_ALPHA) * vy_b_meas

        # --- belly-cam lockout detection ---
        # If the Tello reports near-zero velocity but the IMU sees
        # meaningful horizontal accel, the optical-flow sensor has
        # almost certainly lost lock. We require N consecutive ticks
        # so a single noisy sample doesn't flip the confidence flag.
        vel_mag  = math.hypot(vx_b_meas, vy_b_meas)
        accel_mag = math.hypot(ax_mg, ay_mg)
        if vel_mag < LOCKOUT_VEL_CMPS and accel_mag > LOCKOUT_ACCEL_MG:
            self._lockout_count = min(self._lockout_count + 1, 100)
        else:
            self._lockout_count = max(self._lockout_count - 1, 0)
        lockout_now = self._lockout_count >= LOCKOUT_CONSECUTIVE_TICKS

        # --- rotate to world frame and integrate ---
        # theta is the drone's world heading in radians. With our chosen
        # convention (yaw_deg increases clockwise → world is mirrored y)
        # the rotation matrix that takes (body_forward, body_right) into
        # (world_x = +takeoff-forward, world_y = +takeoff-right) is:
        #   [cos θ  sin θ]   (because both heading and y-right are CW
        #   [-sin θ cos θ]    when viewed from above).
        theta = math.radians(_as_float(tele.get("yaw_deg"), 0.0))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx_w_cmps = vx_b * cos_t + vy_b * sin_t
        vy_w_cmps = -vx_b * sin_t + vy_b * cos_t

        # If lockout is suspected, freeze the position integral but keep
        # heading + velocity smoothing fresh so we pick up cleanly when
        # flow lock comes back.
        if not lockout_now:
            dx_m = (vx_w_cmps * dt) / 100.0
            dy_m = (vy_w_cmps * dt) / 100.0
        else:
            dx_m = dy_m = 0.0

        with self._state_lock:
            self._vel_smooth_body = (vx_b, vy_b)
            self._pose_x_m += dx_m
            self._pose_y_m += dy_m
            self._pose_theta_rad = theta
            self._lockout_active = lockout_now
            self._pose_confidence = "low" if lockout_now else "ok"
            self._trajectory.append(
                (self._pose_x_m, self._pose_y_m, self._pose_confidence)
            )

    # ------------------------------------------------------------------ #
    # render + obstacle loop
    # ------------------------------------------------------------------ #

    def _render_loop(self) -> None:
        period = 1.0 / RENDER_HZ
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        next_deadline = time.monotonic()
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
            try:
                frame = self._get_frame()
            except Exception:
                frame = None

            # Obstacle stamping is gated on being airborne — on the
            # ground the camera is pointed at the floor, not at the
            # room, so any stamping would just be noise.
            if frame is not None and self._was_flying:
                try:
                    self._stamp_from_frame(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mapper obstacle stamping failed: %s", exc)

            try:
                viz = self._render()
            except Exception as exc:  # noqa: BLE001
                logger.warning("mapper render failed: %s", exc)
                continue

            ok, buf = cv2.imencode(".jpg", viz, encode_params)
            if not ok:
                continue
            with self._latest_lock:
                self._latest_jpeg = buf.tobytes()
            self._last_render_ms = int((time.monotonic() - t0) * 1000)

    def _stamp_from_frame(self, frame_bgr: np.ndarray) -> None:
        """Run MiDaS once, stamp the result onto the log-odds grid in
        world coordinates. Fan-shaped: a wedge ahead of the drone at
        the current heading, ± half the camera FOV.

        We don't try to convert MiDaS to metric depth. Instead we just
        record "the central forward patch is currently blocked" or
        "...currently clear", and stamp a coarse fan into the grid at
        the camera bearing. Repeated observations average out.
        """
        # Skip obstacle updates while the position integral is
        # untrusted — stamping at a wrong pose would pollute the grid.
        if self._lockout_active:
            return

        inv = _midas_inverse_depth(frame_bgr)
        if inv is None:
            self._midas_available = False
            return
        self._midas_available = True

        # Same patch + thresholds as ``perception.check_path_clear`` so
        # the two stay consistent. We split the forward third of the
        # frame into three vertical bins (left / centre / right) and
        # stamp per-bin, which gives a richer map than a single yes/no.
        h, w = inv.shape
        norm = (inv - inv.min()) / (inv.ptp() + 1e-6)
        band = norm[h // 4 : 3 * h // 4, :]

        with self._state_lock:
            x0 = self._pose_x_m
            y0 = self._pose_y_m
            theta = self._pose_theta_rad
            pitch_deg = 0.0  # we re-read pitch below from the latest tele

        # Pitch-tilt correction. The forward camera tilts with the
        # drone's pitch (nose-down for forward flight). Pitch in degrees
        # — positive nose-up by the Tello convention — we treat pitch
        # as a vertical offset on the bearing that doesn't affect XY
        # mapping for a small angle, so for now we just log it and skip
        # the projection if pitch is implausibly large. Keeps the math
        # honest without overclaiming we have a 3D model.
        try:
            tele = self._get_tele() or {}
            pitch_deg = _as_float(tele.get("pitch_deg"), 0.0)
        except Exception:
            pass
        if abs(pitch_deg) > 30.0:
            return

        # Build the bin → bearing map.
        bins = 3
        bin_w = w // bins
        fov_rad = math.radians(CAMERA_H_FOV_DEG)
        for i in range(bins):
            patch = band[:, i * bin_w : (i + 1) * bin_w]
            if patch.size == 0:
                continue
            ratio = float((patch > NEAR_THRESHOLD).mean())
            # Bearing for the bin centre: -fov/2 .. +fov/2 across the
            # frame, then add the world heading. Tello camera +x maps
            # to image-right (after the BGR conversion in drone.py), so
            # a higher bin index = world-right of heading = positive
            # CW offset.
            bin_center_norm = (i + 0.5) / bins - 0.5  # -0.5 .. +0.5
            bearing = theta + bin_center_norm * fov_rad
            self._stamp_fan(
                x0, y0, bearing,
                blocked=(ratio > OBSTACLE_MAX_RATIO),
                ratio=ratio,
            )

    def _stamp_fan(
        self, x0: float, y0: float, bearing: float,
        *, blocked: bool, ratio: float,
    ) -> None:
        """Mark cells along a ray from (x0, y0) at ``bearing`` (world
        radians) as free (and the endpoint as occupied if blocked).

        Standard inverse-sensor-model stamping. The strength of each hit
        is scaled by ``ratio`` (how confident the obstacle observation
        was) so a marginal "32% near" reading nudges the grid less than
        a "70% near" reading.
        """
        cx_grid = GRID_CELLS // 2
        cy_grid = GRID_CELLS // 2

        # Step along the ray in 1-cell increments.
        n_steps_free = int(FREE_STAMP_M / CELL_SIZE_M)
        cos_b = math.cos(bearing)
        sin_b = math.sin(bearing)

        with self._grid_lock:
            # Free stamping out to FREE_STAMP_M, capped if blocked at
            # OBS_STAMP_NEAR_M (we stop calling cells "free" past the
            # observed obstacle).
            free_end = (
                OBS_STAMP_NEAR_M if blocked else FREE_STAMP_M
            )
            n_free = max(1, int(free_end / CELL_SIZE_M))
            free_strength = LOG_ODDS_FREE_HIT * (0.6 if blocked else 1.0)
            for s in range(1, n_free + 1):
                d = s * CELL_SIZE_M
                gx = cx_grid + int((x0 + d * cos_b) / CELL_SIZE_M)
                gy = cy_grid + int((y0 + d * sin_b) / CELL_SIZE_M)
                if 0 <= gx < GRID_CELLS and 0 <= gy < GRID_CELLS:
                    self._grid[gy, gx] = max(
                        LOG_ODDS_MIN, self._grid[gy, gx] - free_strength
                    )

            if blocked:
                # Occupied stamping in the OBS_STAMP_NEAR_M..OBS_STAMP_FAR_M
                # band. We don't know the exact distance (MiDaS is non-
                # metric) so we spread the hit across the band.
                n_far = int(OBS_STAMP_FAR_M / CELL_SIZE_M)
                n_near = int(OBS_STAMP_NEAR_M / CELL_SIZE_M)
                hit_strength = LOG_ODDS_OCC_HIT * min(1.0, ratio + 0.2)
                for s in range(n_near, n_far + 1):
                    d = s * CELL_SIZE_M
                    gx = cx_grid + int((x0 + d * cos_b) / CELL_SIZE_M)
                    gy = cy_grid + int((y0 + d * sin_b) / CELL_SIZE_M)
                    if 0 <= gx < GRID_CELLS and 0 <= gy < GRID_CELLS:
                        self._grid[gy, gx] = min(
                            LOG_ODDS_MAX, self._grid[gy, gx] + hit_strength
                        )

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _render(self) -> np.ndarray:
        """Top-down map render. Output is a RENDER_W × RENDER_H BGR
        uint8 image with the takeoff origin at the centre.

        Layers, bottom to top:
          1. Black background
          2. Soft 1 m grid (subtle grey lines)
          3. Free cells (very subtle warm grey)
          4. Occupied cells (red-orange, varies with log-odds strength)
          5. Trajectory polyline (white, faded for the older tail)
          6. Pose arrow (amber, slightly larger when confident)
          7. Origin cross
          8. Top-left status text + scale bar
        """
        canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)

        # Pixel-per-metre — both axes use the same scale because the
        # grid is square and the render is square.
        ppm = RENDER_W / GRID_EXTENT_M
        cx_px = RENDER_W // 2
        cy_px = RENDER_H // 2

        # 1 m grid lines.
        grid_color = (28, 28, 32)
        for i in range(-int(GRID_EXTENT_M / 2), int(GRID_EXTENT_M / 2) + 1):
            x = int(cx_px + i * ppm)
            y = int(cy_px + i * ppm)
            cv2.line(canvas, (x, 0), (x, RENDER_H), grid_color, 1)
            cv2.line(canvas, (0, y), (RENDER_W, y), grid_color, 1)

        # Bigger axis cross at origin.
        axis_color = (48, 48, 54)
        cv2.line(canvas, (cx_px, 0), (cx_px, RENDER_H), axis_color, 1)
        cv2.line(canvas, (0, cy_px), (RENDER_W, cy_px), axis_color, 1)

        # Free + occupied cells. We resize the grid to render dims with
        # nearest-neighbour so cell boundaries stay crisp.
        with self._grid_lock:
            grid = self._grid.copy()
        free_mask = grid < LOG_ODDS_FREE_THRESHOLD
        occ_mask  = grid > LOG_ODDS_OCC_THRESHOLD

        # Build a colour layer at grid resolution then upsample.
        layer = np.zeros((GRID_CELLS, GRID_CELLS, 3), dtype=np.uint8)
        # Free: subtle warm grey on the BGR canvas.
        layer[free_mask] = (32, 36, 40)
        # Occupied: amber-red gradient by log-odds strength.
        occ_strength = np.clip(
            (grid - LOG_ODDS_OCC_THRESHOLD)
            / (LOG_ODDS_MAX - LOG_ODDS_OCC_THRESHOLD + 1e-6),
            0.0, 1.0,
        )
        # BGR amber→red interpolation: low strength is amber (10, 160, 235),
        # full strength is bright red (28, 28, 235). Pre-mix and assign.
        r = (235 + (235 - 235) * occ_strength[occ_mask]).astype(np.uint8)
        g = (160 + (28 - 160) * occ_strength[occ_mask]).astype(np.uint8)
        b = (10  + (28 - 10)  * occ_strength[occ_mask]).astype(np.uint8)
        layer[occ_mask, 0] = b
        layer[occ_mask, 1] = g
        layer[occ_mask, 2] = r

        # Upsample and OR onto the canvas where the layer has content.
        layer_up = cv2.resize(
            layer, (RENDER_W, RENDER_H), interpolation=cv2.INTER_NEAREST,
        )
        layer_has = layer_up.any(axis=2)
        canvas[layer_has] = layer_up[layer_has]

        # Trajectory polyline.
        with self._state_lock:
            pose_x = self._pose_x_m
            pose_y = self._pose_y_m
            pose_t = self._pose_theta_rad
            pose_conf = self._pose_confidence
            traj = list(self._trajectory)

        if len(traj) >= 2:
            pts = []
            for (x_m, y_m, _conf) in traj:
                px = int(cx_px + x_m * ppm)
                py = int(cy_px + y_m * ppm)
                pts.append((px, py))
            # White line, slightly thicker so it pops over the cells.
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_arr], False, (235, 235, 235), 2, cv2.LINE_AA)

        # Origin marker (subtle cross).
        cv2.drawMarker(canvas, (cx_px, cy_px), (150, 150, 150),
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

        # Pose arrow — amber. Smaller / dim when confidence is low.
        px = int(cx_px + pose_x * ppm)
        py = int(cy_px + pose_y * ppm)
        size = 14 if pose_conf == "ok" else 10
        color = (10, 160, 235) if pose_conf == "ok" else (40, 110, 160)
        nose_x = int(px + size * math.cos(pose_t))
        nose_y = int(py + size * math.sin(pose_t))
        # Triangle: nose + two tail points behind the centre.
        tail_a_x = int(px + (size * 0.6) * math.cos(pose_t + 2.5))
        tail_a_y = int(py + (size * 0.6) * math.sin(pose_t + 2.5))
        tail_b_x = int(px + (size * 0.6) * math.cos(pose_t - 2.5))
        tail_b_y = int(py + (size * 0.6) * math.sin(pose_t - 2.5))
        pts = np.array([
            [nose_x, nose_y],
            [tail_a_x, tail_a_y],
            [tail_b_x, tail_b_y],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)

        # Status / scale strip top-left.
        _draw_text(canvas, "Map · dead-reckon", (12, 22), 0.55, 1)
        pose_label = (
            f"x={pose_x:+.2f} m  y={pose_y:+.2f} m  "
            f"θ={math.degrees(pose_t):+.0f}°"
        )
        _draw_text(canvas, pose_label, (12, 42), 0.45, 1, faint=True)
        if pose_conf == "low":
            _draw_text(canvas, "FLOW LOCKOUT — pose drift suspected",
                       (12, RENDER_H - 14), 0.45, 1, color=(60, 60, 235))

        # Scale bar: 1 m in the bottom-right corner.
        bar_len = int(1.0 * ppm)
        bar_x = RENDER_W - bar_len - 20
        bar_y = RENDER_H - 20
        cv2.line(canvas, (bar_x, bar_y), (bar_x + bar_len, bar_y),
                 (200, 200, 200), 2, cv2.LINE_AA)
        cv2.line(canvas, (bar_x, bar_y - 4), (bar_x, bar_y + 4),
                 (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (bar_x + bar_len, bar_y - 4),
                 (bar_x + bar_len, bar_y + 4), (200, 200, 200), 1, cv2.LINE_AA)
        _draw_text(canvas, "1 m", (bar_x + bar_len // 2 - 10, bar_y - 8),
                   0.4, 1, faint=True)

        return canvas

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _pose_dict(self) -> dict[str, float]:
        return {
            "x_m": round(self._pose_x_m, 3),
            "y_m": round(self._pose_y_m, 3),
            "theta_rad": round(self._pose_theta_rad, 3),
            "theta_deg": round(math.degrees(self._pose_theta_rad), 1),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _as_float(value: Any, default: float) -> float:
    """Coerce a telemetry value to a finite float, falling back if it's
    None / non-numeric / NaN."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    thickness: int,
    *,
    faint: bool = False,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Anti-aliased text with a single-pixel drop-shadow for legibility
    against the map's mixed dark/coloured cells. Matches the style used
    by :mod:`depth_stream`."""
    cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    if color is not None:
        fg = color
    else:
        fg = (200, 200, 200) if faint else (240, 240, 240)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg,
                thickness, cv2.LINE_AA)
