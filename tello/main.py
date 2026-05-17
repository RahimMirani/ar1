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

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load tello/.env before importing modules that read API keys at import.
load_dotenv(Path(__file__).resolve().parent / ".env")

from drone import Drone, VALID_FLIP_DIRECTIONS  # noqa: E402
from events import bus  # noqa: E402
from vision import analyze_frame  # noqa: E402
from audio import monitor as audio_monitor  # noqa: E402
from perception import OpticalFlowWatchdog, check_path_clear  # noqa: E402
from depth_stream import DEPTH_HZ, DepthStream  # noqa: E402
from agent import (  # noqa: E402
    configure as agent_configure,
    mission_state,
    is_busy as agent_is_busy,
    run_mission,
)
from notifier import run_notifier_loop, latest_incident  # noqa: E402

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

# Optical-flow watchdog. Auto-starts on takeoff (set_velocity callable is
# the documented fire-and-forget RC path so the watchdog never goes
# through the SDK command lock).
perception_watchdog = OpticalFlowWatchdog(
    get_frame=drone.get_frame,
    drone_stop=drone.stop_velocity,
)

# Live MiDaS visualisation. Pure render thread — never touches the drone
# beyond pulling the latest frame. Start/stop is operator-controlled via
# the Depth view toggle in the console.
depth_stream = DepthStream(get_frame=drone.get_frame)

# Inject the live drone + watchdog into agent.py so its @function_tool
# callables can drive them.
agent_configure(drone, perception_watchdog)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("starting up tello dashboard")
    # Producers in background threads (audio capture, perception watchdog,
    # agent worker) publish via bus.publish_threadsafe, which needs the
    # running loop.
    bus.attach_loop(asyncio.get_running_loop())
    bridge_task   = asyncio.create_task(_audio_alarm_to_agent_bridge())
    notifier_task = asyncio.create_task(run_notifier_loop())
    try:
        yield
    finally:
        logger.info("shutting down: closing drone")
        for task in (bridge_task, notifier_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        depth_stream.stop()
        drone.close()


async def _audio_alarm_to_agent_bridge() -> None:
    """Auto-trigger the agent when the audio detector (or the Simulate
    button) emits an ``audio_alarm`` with state ``alarm``.

    This is the production handoff between Phase B and Phase C. We gate
    on three conditions:

    * the drone is connected (otherwise takeoff is guaranteed to fail);
    * the agent isn't already running (single-flight);
    * the alarm event is a *transition into* ``alarm`` — we ignore
      repeats so a sustained tone doesn't restart the mission.
    """
    queue = bus.subscribe()
    last_state: str | None = None
    try:
        while True:
            ev = await queue.get()
            if ev.get("type") != "audio_alarm":
                continue
            state = ev.get("state")
            transitioned = state == "alarm" and last_state != "alarm"
            last_state = state
            if not transitioned:
                continue
            if not drone.snapshot().connected:
                logger.info("audio alarm received but drone is not connected")
                await bus.publish(
                    {
                        "type": "agent_skipped",
                        "reason": "drone not connected",
                        "trigger": f"audio:{ev.get('source', '?')}",
                    }
                )
                continue
            if agent_is_busy():
                logger.info("audio alarm received but agent is busy")
                await bus.publish(
                    {
                        "type": "agent_skipped",
                        "reason": "agent already running",
                        "trigger": f"audio:{ev.get('source', '?')}",
                    }
                )
                continue
            trigger = f"audio:{ev.get('source', '?')}"
            logger.info("audio alarm -> auto-triggering agent (%s)", trigger)
            asyncio.create_task(_run_mission_task(trigger))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("audio->agent bridge crashed: %s", exc)
    finally:
        bus.unsubscribe(queue)


app = FastAPI(title="FireDrone Tello Dashboard", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Static
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard", include_in_schema=False)
async def dispatcher_dashboard() -> FileResponse:
    """Read-only firefighter-facing real-time view of the current mission.

    Shares the same ``/ws/events`` event stream and ``/video.mjpg`` as
    the operator console, but the page itself never sends commands.
    """
    return FileResponse(STATIC_DIR / "dashboard.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Event bus WebSocket — fan-out for vision/audio/agent/notifier events
# --------------------------------------------------------------------------- #


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Subscribe to the structured event bus.

    Both the operator console and the dispatcher dashboard connect here
    and receive the same stream (vision results, audio alarms, agent
    reasoning, incidents, etc.). One subscriber queue per connection.
    """
    await websocket.accept()
    queue = bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("events websocket closed")
    except Exception as exc:
        logger.warning("events websocket error: %s", exc)
    finally:
        bus.unsubscribe(queue)


# --------------------------------------------------------------------------- #
# Vision — one-shot "analyze current view"
# --------------------------------------------------------------------------- #


@app.post("/api/vision/analyze")
async def api_vision_analyze() -> dict[str, Any]:
    """Grab the most recent frame and run it through the vision model.

    Result is also broadcast on ``/ws/events`` so the dispatcher
    dashboard receives the same payload without an extra round-trip.
    """
    frame = drone.get_frame()
    if frame is None:
        return {"error": "no frame available — connect the drone and start the stream"}

    try:
        result = await asyncio.to_thread(analyze_frame, frame)
    except Exception as exc:
        msg = f"vision analyze failed: {exc}"
        logger.warning(msg)
        return {"error": msg}

    payload = result.to_dict()
    await bus.publish({"type": "vision_result", "source": "manual", **payload})
    return payload


# --------------------------------------------------------------------------- #
# Audio — smoke-alarm detector
# --------------------------------------------------------------------------- #


@app.post("/api/audio/start")
async def api_audio_start() -> dict[str, Any]:
    status = await asyncio.to_thread(audio_monitor.start)
    return status.to_dict()


@app.post("/api/audio/stop")
async def api_audio_stop() -> dict[str, Any]:
    status = await asyncio.to_thread(audio_monitor.stop)
    return status.to_dict()


@app.get("/api/audio/status")
async def api_audio_status() -> dict[str, Any]:
    return audio_monitor.status().to_dict()


@app.post("/api/audio/simulate")
async def api_audio_simulate() -> dict[str, Any]:
    """Manually fire an audio_alarm event for ~4 s.

    Lets you exercise the agent's auto-trigger without holding a smoke
    alarm next to the mic. The downstream code path is identical to a
    real detection.
    """
    return await asyncio.to_thread(audio_monitor.simulate_alarm)


# --------------------------------------------------------------------------- #
# Perception — depth check + optical-flow watchdog
# --------------------------------------------------------------------------- #


@app.post("/api/perception/start")
async def api_perception_start() -> dict[str, Any]:
    return (await asyncio.to_thread(perception_watchdog.start)).to_dict()


@app.post("/api/perception/stop")
async def api_perception_stop() -> dict[str, Any]:
    return (await asyncio.to_thread(perception_watchdog.stop)).to_dict()


@app.get("/api/perception/status")
async def api_perception_status() -> dict[str, Any]:
    return perception_watchdog.status().to_dict()


# --------------------------------------------------------------------------- #
# Depth visualisation stream
# --------------------------------------------------------------------------- #


@app.post("/api/depth/start")
async def api_depth_start() -> dict[str, Any]:
    return (await asyncio.to_thread(depth_stream.start)).to_dict()


@app.post("/api/depth/stop")
async def api_depth_stop() -> dict[str, Any]:
    return (await asyncio.to_thread(depth_stream.stop)).to_dict()


@app.get("/api/depth/status")
async def api_depth_status() -> dict[str, Any]:
    return depth_stream.status().to_dict()


def _depth_mjpeg_generator():
    """Yield JPEG frames from the depth stream as multipart/x-mixed-replace.

    Unlike :func:`_mjpeg_generator` (which encodes per-call from the live
    camera), the depth pipeline pre-encodes on its own 3 Hz worker. We
    just poll the cached buffer at the same cadence. If the stream is
    stopped or hasn't produced its first frame yet, we hold the
    connection open and resume once a buffer becomes available.
    """
    boundary = b"--frame"
    period = 1.0 / DEPTH_HZ
    last_sent_id: int | None = None
    last_payload: bytes | None = None
    while True:
        time.sleep(period)
        payload = depth_stream.latest_jpeg()
        if payload is None:
            # No frame yet (or stream stopped). Re-send the last frame
            # so the <img> tag doesn't blank out during a brief gap.
            if last_payload is None:
                continue
            payload = last_payload
        else:
            if id(payload) == last_sent_id:
                # Worker hasn't produced a new frame yet; skip.
                continue
            last_sent_id = id(payload)
            last_payload = payload

        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
            + b"\r\n"
        )


@app.get("/depth.mjpg")
async def depth_mjpg() -> StreamingResponse:
    return StreamingResponse(
        _depth_mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/perception/check")
async def api_perception_check(direction: str = "forward") -> dict[str, Any]:
    """Run a one-shot depth check on the current frame.

    Useful as a manual smoke test from the operator console (also what
    the agent's ``check_path_clear`` tool calls internally).
    """
    frame = drone.get_frame()
    if frame is None:
        return {"error": "no frame available"}
    result = await asyncio.to_thread(check_path_clear, frame, direction=direction)
    payload = result.to_dict()
    await bus.publish({"type": "perception_check", "source": "manual", **payload})
    return payload


# --------------------------------------------------------------------------- #
# Agent — autonomous mission loop
# --------------------------------------------------------------------------- #


@app.post("/api/agent/start")
async def api_agent_start(trigger: str = "manual") -> dict[str, Any]:
    """Kick off an autonomous mission.

    Returns immediately with the initial mission state; the actual run
    progresses on the FastAPI loop and streams events on ``/ws/events``.
    """
    if agent_is_busy():
        return {"error": "agent already running", **mission_state.to_dict()}
    # Spawn as a task — the endpoint replies right away so the operator
    # console can render the "running" state.
    asyncio.create_task(_run_mission_task(trigger))
    # Give the task a tick to flip state to "starting".
    await asyncio.sleep(0.05)
    return mission_state.to_dict()


@app.get("/api/agent/state")
async def api_agent_state() -> dict[str, Any]:
    return mission_state.to_dict()


async def _run_mission_task(trigger: str) -> None:
    try:
        await run_mission(trigger)
    except Exception as exc:
        logger.exception("mission task crashed: %s", exc)


# --------------------------------------------------------------------------- #
# Incidents — the most recent notifier-emitted incident
# --------------------------------------------------------------------------- #


@app.get("/api/incidents/latest")
async def api_incident_latest() -> dict[str, Any]:
    inc = latest_incident()
    if inc is None:
        return {"incident": None}
    return {"incident": inc}


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
        # Reactive safety: start the optical-flow watchdog whenever the
        # drone is in the air. Idempotent; the watchdog manages its own
        # thread lifecycle.
        await asyncio.to_thread(perception_watchdog.start)
    elif action == "land":
        await asyncio.to_thread(perception_watchdog.stop)
        await asyncio.to_thread(drone.land)
    elif action == "emergency":
        await asyncio.to_thread(perception_watchdog.stop)
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
