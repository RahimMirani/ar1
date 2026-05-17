"""Agent skeleton — the autonomous fire-response loop.

Wires the Drone, vision and perception modules to the OpenAI Agents SDK
and exposes a single :func:`run_mission` coroutine that runs the
take-off / sweep / inspect / decide / land / report cycle on its own.

The agent runs in three constrained ways:

* **One mission at a time.** A process-wide lock prevents concurrent
  ``run_mission`` calls. The audio auto-trigger and the manual
  "Activate agent" button both go through the same gate.
* **Hard wall-clock cap.** ``MISSION_BUDGET_SEC`` (default 180 s) is
  enforced via ``asyncio.wait_for`` around the streamed run. On timeout
  we force-land and emit an error verdict — the drone never just hovers.
* **Single verdict.** The agent must call ``report_finding`` exactly
  once. If it terminates without one, we synthesise an
  ``unknown`` verdict so the dispatcher dashboard still gets an event.

Every meaningful step of the loop is published on the event bus, so
the operator console and the dispatcher dashboard can render the
mission as it unfolds:

* ``agent_state``      idle -> running -> done / error
* ``agent_message``    free-form assistant text between tool calls
* ``agent_tool_call``  ``{tool, args}`` immediately before a tool runs
* ``agent_tool_result````{tool, result}`` immediately after
* ``agent_finding``    final verdict + reasons (terminal)

Phase E's notifier consumes ``agent_finding`` and emits the incident.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agents import Agent, Runner, function_tool

from events import bus
from vision import analyze_frame
from perception import check_path_clear as perception_check

logger = logging.getLogger("tello.agent")

# Plain module-level constants. The agent model is what does the
# reasoning + tool-calling for the autonomous mission; gpt-4o is the
# best trade-off of tool-call reliability and latency we have.
#
# Mission budget + max-turns are sized for *thorough exploratory*
# missions. The expected pattern is takeoff + 4-5 (move, analyze,
# rotate, analyze) sequences + land + report, which lands around
# 24-32 LLM turns and 120-200 s of wall clock. 240 s and 36 turns
# give the agent room to cover the whole 5 m radius without letting
# it wander indefinitely.
AGENT_MODEL = "gpt-4o"
MISSION_BUDGET_SEC = float(os.getenv("FIREDRONE_AGENT_BUDGET_SEC", "240"))
MAX_TURNS = int(os.getenv("FIREDRONE_AGENT_MAX_TURNS", "36"))

VALID_VERDICTS = {"real_fire", "false_alarm", "unknown"}

# --------------------------------------------------------------------------- #
# Module-level wiring (set by main.py at startup via configure())
# --------------------------------------------------------------------------- #

_drone = None  # type: ignore[assignment]
_watchdog = None  # type: ignore[assignment]
_mapper = None  # type: ignore[assignment]


def configure(drone, watchdog, mapper=None) -> None:
    """Hook up the live drone + watchdog + mapper used by the @function_tool
    callables. ``mapper`` is optional so existing call sites that haven't
    been updated yet still work — the pose/map tools degrade gracefully
    to "mapper not configured" when it's absent."""
    global _drone, _watchdog, _mapper
    _drone = drone
    _watchdog = watchdog
    _mapper = mapper


# --------------------------------------------------------------------------- #
# Mission state
# --------------------------------------------------------------------------- #


@dataclass
class MissionState:
    mission_id: str | None = None
    state: str = "idle"  # idle | starting | running | done | error
    trigger: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    verdict: str | None = None
    summary: str | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


mission_state = MissionState()
_mission_lock = asyncio.Lock()

# Per-mission scratch: filled by the @function_tool callables, read by
# run_mission to assemble the final state. We attach via contextvars or
# (here) a simple module-level mutable: only one mission runs at a time
# so a single dict is safe.
_current: dict[str, Any] = {}


def _emit_state(state: str, **extra: Any) -> None:
    mission_state.state = state
    payload = {"type": "agent_state", "state": state, "mission_id": mission_state.mission_id}
    payload.update(extra)
    bus.publish_threadsafe(payload)


def _emit_tool_call(tool: str, args: dict[str, Any]) -> None:
    mission_state.transcript.append({"kind": "tool_call", "tool": tool, "args": args, "ts": time.time()})
    bus.publish_threadsafe(
        {
            "type": "agent_tool_call",
            "mission_id": mission_state.mission_id,
            "tool": tool,
            "args": args,
        }
    )


def _emit_tool_result(tool: str, result: str) -> None:
    mission_state.transcript.append(
        {"kind": "tool_result", "tool": tool, "result": result, "ts": time.time()}
    )
    bus.publish_threadsafe(
        {
            "type": "agent_tool_result",
            "mission_id": mission_state.mission_id,
            "tool": tool,
            "result": result,
        }
    )


def _emit_message(content: str) -> None:
    if not content:
        return
    mission_state.transcript.append({"kind": "message", "content": content, "ts": time.time()})
    bus.publish_threadsafe(
        {
            "type": "agent_message",
            "mission_id": mission_state.mission_id,
            "content": content,
        }
    )


# --------------------------------------------------------------------------- #
# Tools — every one wraps an existing Drone / vision / perception method.
# We never call djitellopy directly here; link-safety invariants stay in
# tello/drone.py.
# --------------------------------------------------------------------------- #


def _tool(name: str, args: dict[str, Any]) -> Callable[[Callable[[], str]], str]:
    """Internal helper: wrap a tool body with bus emit + error handling."""
    def runner(body: Callable[[], str]) -> str:
        _emit_tool_call(name, args)
        try:
            result = body()
        except Exception as exc:
            result = f"ERROR: {exc}"
            logger.warning("tool %s failed: %s", name, exc)
        _emit_tool_result(name, result)
        return result
    return runner


@function_tool
def takeoff() -> str:
    """Take off and climb to the standard patrol altitude (~1.5 m AGL).
    Must be called once before any motion. The obstacle watchdog
    auto-arms on takeoff."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        _drone.takeoff()
        if _watchdog is not None:
            _watchdog.start()
        # Tello's default takeoff lifts to ~80-100 cm. Add ~70 cm so
        # the drone is consistently around 1.5 m for the rest of the
        # patrol — high enough to see over couches, counters, and
        # most floor clutter, low enough to stay clear of light
        # fixtures and ceiling fans in a normal home. We do this in
        # code (not in the prompt) so it always happens and doesn't
        # cost the agent a planning turn.
        try:
            _drone.move("up", 70)
        except Exception as exc:
            logger.warning("post-takeoff climb to patrol altitude failed: %s", exc)
        return "OK - airborne at ~1.5 m patrol altitude; watchdog armed."
    return _tool("takeoff", {})(body)


@function_tool
def land() -> str:
    """Land the drone safely. Stops the obstacle watchdog automatically."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        if _watchdog is not None:
            _watchdog.stop()
        _drone.land()
        return "OK - drone has landed."
    return _tool("land", {})(body)


@function_tool
def hover() -> str:
    """Halt all in-progress motion immediately. Use as a defensive action
    whenever you're unsure what's around."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        _drone.stop_velocity()
        return "OK - velocity zeroed."
    return _tool("hover", {})(body)


@function_tool
def rotate(degrees: int) -> str:
    """Rotate in place. Positive degrees = clockwise (right); negative =
    counter-clockwise (left). Range: -180..180. Useful for sweeping the
    room without flying anywhere."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        deg = max(-180, min(180, int(degrees)))
        if deg == 0:
            return "OK - 0 deg requested, no-op."
        direction = "cw" if deg > 0 else "ccw"
        _drone.rotate(direction, abs(deg))
        return f"OK - rotated {direction} by {abs(deg)} deg."
    return _tool("rotate", {"degrees": degrees})(body)


@function_tool
def move(direction: str, distance_cm: int) -> str:
    """Move in a discrete step. For lateral directions this *also* runs
    a depth check first and refuses if the path is blocked.

    direction: 'forward' | 'back' | 'left' | 'right' | 'up' | 'down'.
    distance_cm: 20-200 — bigger steps cover the room faster; pick
    100-200 for roaming, 20-60 for fine inspection."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        d = direction.strip().lower()
        if d not in {"forward", "back", "left", "right", "up", "down"}:
            return f"ERROR: bad direction {direction!r}"
        cm = max(20, min(200, int(distance_cm)))

        # Floor guard: refuse "down" when we're already close to the
        # ground. tof_cm is the downward time-of-flight reading on the
        # Tello and is reliable below ~120 cm. The cap below leaves a
        # ~20-30 cm safety margin under the worst-case step.
        if d == "down":
            tof = _drone.snapshot().telemetry.get("tof_cm")
            if isinstance(tof, (int, float)) and tof - cm < 40:
                return (
                    f"REFUSED: moving down {cm} cm would put altitude "
                    f"below safe floor (current tof {tof} cm). "
                    "Stay at this altitude or go up."
                )

        # Depth check on motion that brings new scenery in front of the
        # camera. (Back/up/down all show the same scene the camera was
        # already pointing at, so a MiDaS check is uninformative for
        # those.) We publish the result either way so the operator
        # console can show *why* a refusal happened, not just the tool
        # error string.
        if d in {"forward", "left", "right"}:
            frame = _drone.get_frame()
            if frame is not None:
                check = perception_check(frame, direction=d)
                bus.publish_threadsafe(
                    {"type": "perception_check", "source": "move_tool",
                     "mission_id": mission_state.mission_id, **check.to_dict()}
                )
                if check.available and not check.clear:
                    return (
                        f"REFUSED: depth check says {d} is blocked "
                        f"(obstacle ratio {check.obstacle_ratio:.0%}). "
                        "Try rotating, or use `move(\"back\", ...)` to "
                        "back away — back never needs a depth check."
                    )
        _drone.move(d, cm)
        return f"OK - moved {d} {cm} cm."
    return _tool("move", {"direction": direction, "distance_cm": distance_cm})(body)


@function_tool
def analyze_view(prompt: str = "") -> str:
    """Capture the current camera frame and ask the vision model to look
    for fire or smoke. The full result (including the thumbnail) is
    streamed to the operator console + dispatcher dashboard. Returns a
    short text summary for the agent to reason over.

    prompt: optional override. Default question is 'Analyze this camera
    frame for fire or smoke.'"""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        frame = _drone.get_frame()
        if frame is None:
            return "ERROR: no frame available (is the video stream up?)"
        result = analyze_frame(frame, prompt=prompt or None)
        payload = result.to_dict()
        bus.publish_threadsafe(
            {"type": "vision_result", "source": "agent",
             "mission_id": mission_state.mission_id, **payload}
        )
        # Evidence ledger (without the bulky thumbnail string) for the final
        # incident payload.
        evidence_entry = {k: v for k, v in payload.items() if k != "thumbnail_b64"}
        evidence_entry["thumbnail_b64"] = payload.get("thumbnail_b64")
        mission_state.evidence.append(evidence_entry)
        summary = (
            f"severity={result.severity}; "
            f"fire_visible={result.fire_visible}; smoke_visible={result.smoke_visible}; "
            f"confidence={result.confidence:.2f}; "
            f"description='{result.description}'"
        )
        return summary
    return _tool("analyze_view", {"prompt": prompt})(body)


@function_tool
def check_path_clear(direction: str = "forward") -> str:
    """Run an on-demand MiDaS depth check in the given direction.

    direction: 'forward' | 'left' | 'right' | 'up' | 'down'."""
    def body() -> str:
        if _drone is None:
            return "ERROR: drone not configured"
        frame = _drone.get_frame()
        if frame is None:
            return "ERROR: no frame"
        check = perception_check(frame, direction=direction)
        bus.publish_threadsafe(
            {"type": "perception_check", "source": "agent",
             "mission_id": mission_state.mission_id, **check.to_dict()}
        )
        if not check.available:
            return "WARN: MiDaS unavailable - proceed with caution."
        return f"{check.reason} (latency {check.latency_ms} ms)"
    return _tool("check_path_clear", {"direction": direction})(body)


@function_tool
def report_finding(verdict: str, summary: str, reasons: list[str]) -> str:
    """Submit the agent's final verdict. TERMINAL — call exactly once.

    verdict: 'real_fire' or 'false_alarm' (or 'unknown' if you genuinely
    cannot tell).
    summary: 1-2 sentence description of what you found.
    reasons: 2-5 short bullet phrases backing the verdict.

    The finding is sent to the notifier, which publishes an incident
    event to the dispatcher dashboard and (if configured) posts a
    webhook to the simulated fire-department endpoint."""
    def body() -> str:
        v = verdict.strip().lower()
        if v not in VALID_VERDICTS:
            return f"ERROR: verdict must be one of {sorted(VALID_VERDICTS)}"

        # Soft enforcement: clearing an alarm requires *coverage*, not
        # just a glance. Real fires can still be reported quickly with
        # sparse evidence (any single severity="high" analyze_view
        # capture is enough for the model to justify real_fire), but
        # false_alarm must be backed by analyze_view results from
        # multiple vantage points across the room. Two clean captures
        # from the same spot are not enough to clear a fire alarm.
        if v == "false_alarm" and len(mission_state.evidence) < 4:
            return (
                "ERROR: false_alarm requires at least 4 analyze_view "
                f"captures from distinct positions; you have "
                f"{len(mission_state.evidence)}. Move to a new "
                "position, call analyze_view there, then retry "
                "report_finding."
            )

        mission_state.verdict = v
        mission_state.summary = summary
        mission_state.reasons = list(reasons)
        bus.publish_threadsafe(
            {
                "type": "agent_finding",
                "mission_id": mission_state.mission_id,
                "verdict": v,
                "summary": summary,
                "reasons": list(reasons),
                "evidence_count": len(mission_state.evidence),
            }
        )
        return (
            "OK - finding submitted. End the mission now: call land() if you "
            "haven't already, then return a short closing message."
        )
    return _tool("report_finding", {"verdict": verdict, "summary": summary,
                                    "reasons": list(reasons)})(body)


# --------------------------------------------------------------------------- #
# Agent definition + system prompt
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """You are FireDroneAgent — an autonomous fire-response drone
investigating an alarm. The drone is a DJI Tello operating indoors in a home or
small office.

Treat this like an actual search, not a stationary check: don't just rotate
where you are — *go look*. Your operational radius is about 5 metres from the
takeoff point. Visit **at least four distinct positions** in the space and
inspect each from 2-3 angles before submitting a verdict. The point of the
mission is to confidently say "I checked everywhere reasonable and saw X" —
that's only credible if you actually moved through the room.

Mission shape (a template — adapt to what the space looks like):

  1. `takeoff()`. The takeoff tool automatically climbs to ~1.5 m AGL, so
     you start the mission already at patrol altitude. You should not need
     to call `move("up", ...)` at all in a normal mission — leave altitude
     alone and focus on lateral coverage.
  2. **Initial scan from takeoff position.** Call `analyze_view()`. If it
     comes back fire_visible=true with confidence >= 0.6, fast-track to
     report — do one corroborating capture from a different angle, then land
     and report real_fire.
  3. **Patrol outward.** Pick a direction (start with what the camera sees).
     Call `check_path_clear("forward")` — if clear, `move("forward", 150-200)`.
     Otherwise rotate 45-90 deg and try again. At the new position call
     `analyze_view()`. Then `rotate(90)` and analyze again to cover the side
     you couldn't see from the previous spot.
  4. **Visit at least three more distinct positions.** Good shapes:
       * Diamond: forward 180 -> rotate 90 -> forward 150 -> rotate 90 ->
                  forward 180 -> rotate 90 -> forward 150 (back near start).
       * Hallway sweep: forward 200, analyze, forward 200, analyze, rotate
                        180, forward 200, analyze, forward 200, analyze.
       * Wide arc: forward 150, rotate 45, forward 150, rotate 45,
                   forward 150 — fans out across the open space.
     Always `check_path_clear` before a forward step. If REFUSED, treat it
     as useful information — that direction has a wall or obstacle. Pick
     another. **Do not give up on lateral motion** — covering ground IS the
     mission. If a `move("forward", ...)` call is REFUSED, your next move
     should be either `rotate(90)` and re-check, or `move("back", 100-150)`
     to retreat into clear space, or `move("left", ...)` / `move("right",
     ...)`. **Vertical motion is not exploration** — `move("up"/"down", ...)`
     does not count toward your distinct-position requirement.

     In a typical empty home or office, the depth check will return CLEAR
     for forward most of the time. That is your *green light* to push
     forward with confidence, not a hint to stop early. Open rooms and
     hallways should get long forward moves (180-200 cm), not 60 cm hops.
     Save the small steps for tight spots where you're inspecting a
     specific area.
  5. **Return**: rotate roughly back toward your start so the operator's
     orientation makes sense, then `land()`.
  6. Call `report_finding(verdict, summary, reasons)` exactly once. The
     report_finding tool will REFUSE a false_alarm verdict if you have
     fewer than four analyze_view captures on record — that's the
     "you actually patrolled" check.
  7. One short closing sentence is enough.

Direction reference:

  * `forward` / `left` / `right` — go through the MiDaS depth check.
    May be REFUSED if the path is blocked.
  * `back` — always available. Use this to retreat from a refused
    forward, or to back up for a wider view before analyze.
  * `up` / `down` — always available, but altitude changes do not
    fulfil the "visit distinct positions" requirement.

Decision rules:

  * real_fire   — *two or more* `analyze_view` results agree on
                  fire_visible=true with confidence >= 0.55, OR any single
                  result shows severity == "high". The "two or more" rule is
                  the conservative bias: indoor false-positive triggers
                  (red curtains, sunset light, computer screens, candle
                  flames, LED strips) are extremely common, so we want
                  corroboration from different angles before paging the
                  fire department.
  * false_alarm — all captures came back clean, OR you can name the
                  specific innocent thing that fooled the alarm (a candle,
                  a kitchen-screen glow, etc).
  * unknown     — last resort. Only when vision repeatedly errored or you
                  genuinely cannot decide after a full patrol.

Hard rules:

  * NEVER call `move("forward", ...)` without first calling
    `check_path_clear("forward")`. If it refuses, do not retry forward in
    the same orientation — rotate, then re-check, then move.
  * If `move("down", ...)` is REFUSED for low altitude, do not retry going
    down — you're already near the floor. Stay level or go up.
  * If any tool returns `ERROR:` or `REFUSED:`, adapt instead of repeating
    the same call.
  * Conservative on real_fire — your reasons must cite at least two
    *independent visual cues* (ideally from two different positions or
    angles). "I saw orange" alone is not enough.
  * Be specific in your reasons. Good: "orange flicker behind the couch,
    haze hugging the ceiling near the kitchen entrance." Bad: "fire visible
    in the room."

Output style: short and procedural. The operator is watching your reasoning
live in a UI, so think in 1-2 sentence beats between tool calls — what you
plan to do next, and what the last result told you. No bullet lists, no
multi-paragraph plans. Action over deliberation."""


def _build_agent() -> Agent:
    return Agent(
        name="FireDroneAgent",
        instructions=SYSTEM_PROMPT,
        model=AGENT_MODEL,
        tools=[
            takeoff,
            land,
            hover,
            rotate,
            move,
            analyze_view,
            check_path_clear,
            report_finding,
        ],
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _reset_state(trigger: str) -> str:
    mission_state.__init__()  # type: ignore[misc]
    mission_state.mission_id = uuid.uuid4().hex[:10]
    mission_state.state = "starting"
    mission_state.trigger = trigger
    mission_state.started_at = time.time()
    return mission_state.mission_id


def is_busy() -> bool:
    return mission_state.state in {"starting", "running"}


async def run_mission(trigger: str = "manual") -> MissionState:
    """Run one autonomous mission end-to-end. Single-flight via a lock.

    Returns the final ``MissionState`` snapshot. Whether the agent ran
    to a clean verdict, timed out, or crashed, this coroutine always
    attempts to land the drone before returning.
    """
    if _mission_lock.locked():
        raise RuntimeError("a mission is already running")

    async with _mission_lock:
        mid = _reset_state(trigger)
        _emit_state("starting", trigger=trigger)
        logger.info("mission %s start (trigger=%s)", mid, trigger)

        agent = _build_agent()
        user_prompt = (
            "A fire alarm has just been triggered. "
            f"Trigger source: {trigger}. "
            "Patrol the home within ~5 m of the takeoff point. The expected "
            "shape is: takeoff -> at least FOUR different positions, each "
            "with one or two analyze_view captures -> land -> report. The "
            "depth feed will usually say 'forward CLEAR' inside an empty "
            "home; that's permission to keep moving, not a reason to land. "
            "Only short-circuit this pattern if you actually see fire/smoke "
            "with high confidence — in that case wrap up immediately."
        )

        mission_state.state = "running"
        _emit_state("running")

        try:
            run_result = Runner.run_streamed(
                agent,
                input=user_prompt,
                max_turns=MAX_TURNS,
            )

            async def consume() -> None:
                async for event in run_result.stream_events():
                    _handle_stream_event(event)

            await asyncio.wait_for(consume(), timeout=MISSION_BUDGET_SEC)

            if mission_state.verdict is None:
                # Agent forgot to call report_finding — synthesise one.
                mission_state.verdict = "unknown"
                mission_state.summary = (
                    "Agent ended without a verdict — synthesised fallback."
                )
                mission_state.reasons = ["agent did not call report_finding"]
                bus.publish_threadsafe(
                    {
                        "type": "agent_finding",
                        "mission_id": mid,
                        "verdict": "unknown",
                        "summary": mission_state.summary,
                        "reasons": mission_state.reasons,
                        "evidence_count": len(mission_state.evidence),
                        "synthesised": True,
                    }
                )

            mission_state.state = "done"
            mission_state.ended_at = time.time()
            _emit_state("done", verdict=mission_state.verdict)
            logger.info("mission %s done verdict=%s", mid, mission_state.verdict)

        except asyncio.TimeoutError:
            mission_state.error = f"mission exceeded {MISSION_BUDGET_SEC:.0f} s budget"
            await _safe_land()
            mission_state.state = "error"
            mission_state.ended_at = time.time()
            _emit_state("error", error=mission_state.error)
            logger.warning("mission %s timed out", mid)

        except Exception as exc:  # noqa: BLE001
            mission_state.error = f"{type(exc).__name__}: {exc}"
            await _safe_land()
            mission_state.state = "error"
            mission_state.ended_at = time.time()
            _emit_state("error", error=mission_state.error)
            logger.exception("mission %s crashed", mid)

    return mission_state


async def _safe_land() -> None:
    """Best-effort land + watchdog stop, used in error paths."""
    if _drone is None:
        return
    try:
        if _watchdog is not None:
            await asyncio.to_thread(_watchdog.stop)
    except Exception:
        pass
    try:
        await asyncio.to_thread(_drone.land)
    except Exception:
        try:
            await asyncio.to_thread(_drone.emergency)
        except Exception:
            pass


def _handle_stream_event(event: Any) -> None:
    """Translate openai-agents stream events into bus events.

    We only consume a small subset of the stream — enough to render
    'thoughts between tool calls' and the model's closing message in
    the UI. Tool start/end already flow through our ``_tool`` wrappers.
    """
    etype = getattr(event, "type", None)
    if etype != "run_item_stream_event":
        return
    item = getattr(event, "item", None)
    if item is None:
        return
    item_type = getattr(item, "type", None)

    if item_type == "message_output_item":
        # Collect any free-form assistant text from the item.
        raw = getattr(item, "raw_item", None)
        content = _extract_text(raw)
        if content:
            _emit_message(content)


def _extract_text(raw: Any) -> str:
    """Best-effort pull of the text out of a ResponseOutputMessage.

    The openai-agents SDK wraps OpenAI's response objects. We try the
    common shapes and fall back to ``str()`` so a future SDK refactor
    doesn't take the dashboard out.
    """
    if raw is None:
        return ""
    parts = getattr(raw, "content", None)
    if isinstance(parts, str):
        return parts
    if not parts:
        return ""
    out: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            out.append(text)
            continue
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            out.append(part["text"])
    return "\n".join(s for s in out if s).strip()
