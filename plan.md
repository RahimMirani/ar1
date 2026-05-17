# FireDrone — Plan of Action (Tello)

## What we're building

An autonomous fire-response drone agent. When a fire alarm sounds in the
house, the drone arms, flies a quick inspection of the room via its camera,
and decides whether it's a **real fire** or a **false alarm**. If real, it
sends a notification (simulated fire-department alert on the dashboard). If
false, it logs the incident and lands.

The pitch is the **false-alarm filter**: most residential fire-alarm trips
are false, and dispatching a truck is expensive. An autonomous agent that
verifies before escalating is the product.

## Hardware

- **DJI Tello** (original, not EDU). UDP SDK over its own WiFi access point.
- **Laptop** — runs everything (mic, OpenAI agent, vision, dashboard, drone
  control). WiFi NIC joins the Tello's network; USB ethernet provides
  internet for OpenAI calls.
- **USB ethernet adapter** — required, because the Tello WiFi has no
  internet and the original Tello has no station mode.

> Pivoted from BetaFPV Air75 / DroneForge / NimbusOS / RadioMaster. The old
> `firedrone/` folder is preserved as a historical reference but not used.

## Software stack

- **`djitellopy`** — Python SDK for the Tello. Sends commands, receives state
  packets, exposes the H.264 video stream.
- **`drone.py`** (`tello/drone.py`) — the only module allowed to make raw SDK
  calls. Owns connection, telemetry caching, the live RC velocity vector,
  and discrete distance commands for the agent.
- **FastAPI server** (`tello/main.py`) — WebSocket telemetry + control,
  MJPEG video, static dashboard. The browser is the operator interface.
- **Web dashboard** (`tello/static/`) — vanilla HTML/CSS/JS. Live video,
  telemetry, hold-to-fly controls, event log. Will host the agent + audio +
  vision panels in upcoming phases.

## Networking

The Tello broadcasts `TELLO-XXXXXX` (no internet). Laptop's WiFi NIC joins
that for drone control; USB ethernet handles internet for OpenAI. Windows
routes by destination IP — Python code doesn't have to care, just confirm
both `ping 192.168.10.1` (drone) and `ping 1.1.1.1` (internet) succeed.

One Windows-specific gotcha worth remembering: the Tello WiFi must be set
to the **Private** profile, otherwise inbound UDP 8890 / 11111 are silently
dropped even with explicit firewall allow rules.

## Architecture

```
[Mic] ──► audio.py ──┐
                      ▼
                 agent.py ──► OpenAI Agents SDK (reasoning + tool calls)
                  │   ▲
[Tello] ──► drone.py    │   │
   │ camera   │  │   vision.py ──► OpenAI vision (gpt-4o / gpt-4o-mini)
   │ state    │  │
   │ commands ▼  │
   │       main.py (FastAPI: WS + MJPEG + REST)
   │              │
   │     ┌────────┴────────┐
   │     ▼                 ▼
   └──► dashboard      notifier (simulated fire-dept alert)
        (browser)
```

## Tech choices

- Python 3.12, `uv` for env management. Single root `pyproject.toml`.
- `djitellopy` (live), `opencv-python` (frame encoding), `av` (decoder
  patched for low-delay).
- **OpenAI Agents SDK** (`openai-agents`) for the agent loop, tool
  registration, streaming. Replaces the hand-rolled chat-completions loop
  from the original plan — same intent, much less code.
- **OpenAI vision** (`gpt-4o-mini` for routine "look at this frame" calls;
  `gpt-4o` for the final real-vs-false decision) — no local CLIP.
- **Audio**: FFT-based detector for the ~3-4 kHz pulsed pattern of a
  residential smoke alarm. Pure DSP, no ML model. (`sounddevice` for input,
  `numpy` for the FFT.)
- FastAPI + vanilla HTML/CSS/JS dashboard. No build step.
- Notifier: simulated alert banner on the dashboard. Twilio SMS optional.

## What's already done (milestone 1)

Bench-tested and pushed to `main`:

- Tello connects over WiFi; UDP 8890/11111 firewall rules + Private profile
  documented.
- `Drone` wrapper with hold-to-fly **RC velocity** model: WASD / Space /
  Shift / Q / E send instant velocity updates at 20 Hz; emergency stop on
  `Esc`; WebSocket-disconnect → motors cut.
- Live video MJPEG with low-latency PyAV options + RGB→BGR fix; resilient
  H.264 demux (skip corrupt NAL units without killing the worker).
- Discrete `move()` / `rotate()` / `flip()` methods preserved on the
  `Drone` for the agent to call.
- Dashboard: live video, telemetry grid (battery / height / yaw / pitch /
  roll / speeds / temp), connect-disconnect, takeoff / land / flips,
  scrolling event log.

## Plan of action (next)

### Phase A — Vision spine

`tello/vision.py`: `analyze_frame(bgr_frame) -> dict` that JPEG-encodes the
frame, resizes to 768×768, sends to OpenAI vision with a JSON-schema
constrained response, returns
`{fire_visible, smoke_visible, confidence, description}`. Plus a
`smoke_vision.py` script and a dashboard button "Analyze current view"
that displays the thumbnail + JSON.

Goal: validate the OpenAI round-trip with real Tello frames before
anything else is built on top.

### Phase B — Audio trigger (parallel to A)

`tello/audio.py`: `sounddevice.InputStream`, FFT detector, debounced
`fire_alarm_detected` event (require N consecutive positive windows so
random noise doesn't trigger it). Dashboard gets an audio meter + alarm
badge + a "simulate alarm" button for testing.

Goal: reliable alarm detection against YouTube clips played through a
speaker, decoupled from anything that flies.

### Phase C — Agent skeleton

`tello/agent.py`: OpenAI Agents SDK loop with tools:

- `take_off()`, `land()`, `emergency_stop()`, `move(direction, cm)`,
  `rotate(direction, degrees)` — wrap the existing `Drone` methods
- `analyze_current_view()` — wraps `vision.analyze_frame()`
- `report_real_fire(description, severity)` /
  `report_false_alarm(reason)` — terminal tools, end the loop
- `return_and_land()` — terminal tool, safe abort

System prompt frames the agent as an autonomous fire-safety operator.
Hard 90-second flight cap enforced inside `agent.py`. Reasoning streams
to the dashboard via a new WebSocket event type. Manually triggered by a
button at first.

Goal: agent can complete a full inspection from the dashboard button
with props off (or in flight in a cleared room), producing a sensible
classification and tool sequence.

### Phase D — Integrate

Audio event from Phase B fires the Phase C agent. Dashboard shows the
full pipeline live: audio spike → alarm badge → agent active → reasoning
stream → vision thumbnails → drone moves → classification.

### Phase E — Notification + polish

`tello/notifier.py`: a simulated fire-department alert. Dashboard banner
turns red with severity + description on real fire; gray "FALSE ALARM
LOGGED" on no-fire. Demo flow refinement. Twilio SMS as a stretch.

## Demo flow

1. Tello on the floor, laptop on Tello WiFi + USB ethernet, dashboard open.
2. Play a fire-alarm clip through a speaker.
3. Audio detector lights the alarm badge on the dashboard.
4. Agent kicks off; reasoning streams to the dashboard:
   *"Fire alarm detected. Initiating inspection. Examining current view..."*
5. Drone takes off, rotates to inspect (a couple of yaw + short moves in
   the room).
6. Each vision call shows a thumbnail + structured JSON on the dashboard.
7. Agent classifies: real fire → red banner with simulated dispatch
   message; false alarm → gray log entry.
8. Drone lands. Operator `Esc` available throughout as the kill switch.

## Cut-scope priority (if behind)

Drop in this order:

1. Twilio SMS (already an optional stretch)
2. Multi-room patrol → hover-and-rotate in one room
3. Full autonomy → pilot manually via dashboard while the agent operates
   on the video feed (still a compelling demo: vision + reasoning + decision)

**Do not cut**: the dashboard, visible agent reasoning, real-vs-false
decision, simulated notification.

## Safety rules

- `Esc` is the kill switch (`tello.emergency()` cuts motors). Always
  reachable from the dashboard, both as a button and as a key.
- WebSocket-disconnect → motors cut. Already implemented.
- Hard 90-second agent flight cap (enforced inside `agent.py`).
- Battery <15% turns the indicator red on the dashboard; auto-land is not
  enforced in code yet, manual land required.
- Always fly in a cleared room. No obstacle avoidance, no physical RC
  override on the original Tello.
- Have 2-3 charged batteries on hand for the demo (Tello hover ~13 min).
