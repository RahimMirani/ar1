"""FastAPI server: web dashboard for the Tello drone.

Endpoints
---------
* ``GET  /``               → serves the static dashboard
* ``GET  /video.mjpg``     → MJPEG video stream (one client at a time is fine)
* ``WS   /ws/telemetry``   → broadcast telemetry + status @ 5 Hz
* ``WS   /ws/control``     → receives JSON commands from the browser. While
                              a client is connected and the drone is in the
                              air, dropping the WebSocket triggers an
                              emergency motor cut as the only safety net.
* ``POST /api/connect``    → connect to the Tello and start the video stream

Run::

    cd tello
    uv run uvicorn main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from drone import Drone, VALID_FLIP_DIRECTIONS

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tello.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"

TELEMETRY_HZ = 5.0
VIDEO_FPS = 25.0
JPEG_QUALITY = 70


# Single shared drone instance for the process. The dashboard is single-user.
drone = Drone()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("starting up tello dashboard")
    try:
        yield
    finally:
        logger.info("shutting down: closing drone")
        drone.close()


app = FastAPI(title="FireDrone Tello Dashboard", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Static
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Connection endpoint
# --------------------------------------------------------------------------- #


@app.post("/api/connect")
async def api_connect() -> dict[str, Any]:
    """Connect to the Tello and start the video stream. Idempotent.

    The response also carries ``link_diagnostics`` — a dict of bool flags
    confirming that the link-safety monkey-patches applied and the
    background threads are alive. The dashboard surfaces any False entry
    in the event log so a regression here is visible immediately rather
    than during flight.
    """

    def _do() -> dict[str, Any]:
        snap = drone.snapshot()
        if not snap.connected:
            drone.connect()
        if not drone.snapshot().streaming:
            drone.start_stream()
        drone.clear_error()
        return {
            **asdict(drone.snapshot()),
            "link_diagnostics": drone.link_diagnostics(),
        }

    try:
        return await asyncio.to_thread(_do)
    except Exception as exc:
        msg = f"connect failed: {exc}"
        logger.warning(msg)
        drone.set_error(msg)
        return {"error": msg, **asdict(drone.snapshot())}


@app.post("/api/disconnect")
async def api_disconnect() -> dict[str, Any]:
    def _do() -> dict[str, Any]:
        drone.close()
        return asdict(drone.snapshot())

    return await asyncio.to_thread(_do)


# --------------------------------------------------------------------------- #
# Video — MJPEG
# --------------------------------------------------------------------------- #


def _mjpeg_generator():
    """Yield JPEG frames in multipart/x-mixed-replace format."""

    boundary = b"--frame"
    period = 1.0 / VIDEO_FPS
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    last_sent: Optional[bytes] = None
    next_deadline = time.monotonic()

    while True:
        next_deadline += period
        sleep_for = next_deadline - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # We fell behind — resync deadline to now.
            next_deadline = time.monotonic()

        frame = drone.get_frame()
        if frame is None:
            if last_sent is None:
                continue
            payload = last_sent
        else:
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            payload = buf.tobytes()
            last_sent = payload

        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
            + b"\r\n"
        )


@app.get("/video.mjpg")
async def video_mjpg() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# Telemetry WebSocket
# --------------------------------------------------------------------------- #


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    period = 1.0 / TELEMETRY_HZ
    try:
        while True:
            await websocket.send_json(asdict(drone.snapshot()))
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        logger.info("telemetry websocket closed")
    except Exception as exc:
        logger.warning("telemetry websocket error: %s", exc)


# --------------------------------------------------------------------------- #
# Control WebSocket — emergency on disconnect
# --------------------------------------------------------------------------- #


async def _execute_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a single control message in a worker thread.

    Live motion is done via ``set_velocity``: the dashboard streams updates
    whenever the user's held-key combination changes, and the drone's RC
    background thread keeps forwarding the vector at 20 Hz.
    """

    action = cmd.get("action")

    if action == "ping":
        return {"ok": True, "action": "ping"}

    if action == "takeoff":
        await asyncio.to_thread(drone.takeoff)
    elif action == "land":
        await asyncio.to_thread(drone.land)
    elif action == "emergency":
        await asyncio.to_thread(drone.emergency)
    elif action == "set_velocity":
        lr  = int(cmd.get("lr", 0))
        fb  = int(cmd.get("fb", 0))
        ud  = int(cmd.get("ud", 0))
        yaw = int(cmd.get("yaw", 0))
        await asyncio.to_thread(drone.set_velocity, lr, fb, ud, yaw)
        # Quiet response — these arrive at high rate; do not spam the log.
        return {"ok": True, "action": "set_velocity", "silent": True}
    elif action == "stop_velocity":
        await asyncio.to_thread(drone.stop_velocity)
    elif action == "flip":
        direction = cmd.get("direction", "")
        if direction not in VALID_FLIP_DIRECTIONS:
            return {"ok": False, "error": f"bad flip: {direction!r}"}
        await asyncio.to_thread(drone.flip, direction)
    else:
        return {"ok": False, "error": f"unknown action: {action!r}"}

    return {"ok": True, "action": action, "status": drone.snapshot().last_status}


@app.websocket("/ws/control")
async def ws_control(websocket: WebSocket) -> None:
    await websocket.accept()
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info("control websocket connected from %s", client)

    try:
        while True:
            cmd = await websocket.receive_json()
            try:
                response = await _execute_command(cmd)
            except Exception as exc:
                logger.warning("command failed: %s -> %s", cmd, exc)
                response = {"ok": False, "error": str(exc), "command": cmd}
                drone.set_error(str(exc))
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("control websocket dropped from %s", client)
    except Exception as exc:
        logger.warning("control websocket error: %s", exc)
    finally:
        await _safety_emergency_on_disconnect()


async def _safety_emergency_on_disconnect() -> None:
    """If the drone is in the air when the control WS drops, cut motors.

    This is the only automatic safety we have in milestone 1: the browser is
    the operator, so losing the browser means losing the operator. We don't
    try a controlled land — we just stop the motors. The drone falls a short
    distance and is safer than drifting blind.
    """

    snap = drone.snapshot()
    if not snap.flying:
        return
    logger.warning("control disconnected mid-flight — issuing emergency stop")
    await asyncio.to_thread(drone.emergency)
