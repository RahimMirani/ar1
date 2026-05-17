# FireDrone — Plan of Action (Tello)

## What we're building

An autonomous fire-response drone agent. When a fire alarm sounds in the
house, the drone arms, flies a quick inspection of the room via its camera,
and decides whether it's a **real fire** or a **false alarm**. Either way
a notification is sent:

- **Real fire** → simulated fire-department alert with severity,
  description, and the thumbnail of the deciding camera frame.
- **False alarm** → "alarm cleared" notification with the **reason**
  (cooking steam, sunlight, sensor fault, candle, etc.), the full
  agent-authored explanation of what it observed, and the supporting
  thumbnails. The false-alarm notification is just as detailed as the
  real-fire one — it's the audit trail.

Then the drone lands.

The pitch is the **false-alarm filter**: most residential fire-alarm trips
are false, and dispatching a truck is expensive. An autonomous agent that
verifies before escalating, *and explains its reasoning either way*, is the
product.

## Hardware

- **DJI Tello** (original, not EDU). UDP SDK over its own WiFi access point.
- **Laptop** — runs everything (mic, OpenAI agent, vision, dashboard, drone
  control). WiFi NIC joins the Tello's network; USB ethernet provides
  internet for OpenAI calls.
- **USB ethernet adapter** — required, because the Tello WiFi has no
  internet and the original Tello has no station mode.

> Pivoted from BetaFPV Air75 / DroneForge / NimbusOS / RadioMaster. The old
> `firedrone/` code has been removed; this project is Tello-only.

## Software stack

- **`djitellopy`** — Python SDK for the Tello. Sends commands, receives state
  packets, exposes the H.264 video stream.
- **`drone.py`** (`tello/drone.py`) — the only module allowed to make raw SDK
  calls. Owns connection, telemetry caching, the live RC velocity vector,
  and discrete distance commands for the agent.
- **FastAPI server** (`tello/main.py`) — WebSocket telemetry + control,
  MJPEG video, static dashboard. The browser is the operator interface.
- **Operator console** (`/`) — vanilla HTML/CSS/JS, Inter typeface, flat
  dark theme with restrained amber accents. Live video with glass HUD
  insets (callsign + altitude / battery / flight-time, link-feed-airborne
  state, live velocity vector). Topbar mission pill (idle / armed /
  flight / emergency) and indicator dots. Side pane: connection card,
  telemetry card with prominent battery bar, flight card, motion pad with
  hold-to-fly buttons and live velocity readout, flips card. Bottom:
  event log. Will gain alarm / agent-reasoning / vision-thumbnail /
  notification panels in upcoming phases — added as new side-pane cards
  and a banner zone, in the same visual language.
- **Dispatcher dashboard** (`/dashboard`) — separate page served by the
  same FastAPI app, subscribing to the same WebSocket events. This is
  what the **fire department sees in real time** when they receive a
  notification: read-only, no controls, designed for a station screen or
  a firefighter's phone. Same flat dark / amber visual language as the
  operator console but laid out for storytelling at a glance — large
  full-bleed live video, prominent alarm banner, agent reasoning
  streamed in large readable type, vision thumbnails as a filmstrip, and
  the verdict banner taking over half the screen at decision time. Lands
  in Phase E.

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
                  │   ▲   ▲
[Tello] ──► drone.py    │   │   │
   │ camera   │  │     │   vision.py ──► OpenAI vision (gpt-4o / gpt-4o-mini)
   │ state    │  │     │
   │ commands │  │     perception.py ── MiDaS depth + cv2 optical flow (local)
   │          ▼  │
   │       main.py (FastAPI: WS + MJPEG + REST)
   │              │
   │     ┌────────┴────────────────────────┐
   │     ▼                                 ▼
   └──► browser views                  notifier (incident event)
        ├─ /          operator console
        └─ /dashboard dispatcher view (firefighters)
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
- **Perception (local)**: **MiDaS v2.1 Small** monocular depth model
  (~21 MB ONNX) loaded via `cv2.dnn.readNet` — no torch, no transformers,
  ~50-80 ms per inference at 256×256 on CPU. Plus **Farnebäck dense optical
  flow** via `cv2.calcOpticalFlowFarneback` for a reactive watchdog. Both
  run on-device. No new Python deps; OpenCV is already installed.
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
- Operator console: flat dark UI in Inter, amber-on-active accents.
  Topbar with mission-state pill + link/feed/air indicator dots +
  emergency-stop. Video pane with three glass HUD insets (callsign +
  battery / altitude / time; live link-feed-airborne state; live
  velocity vector). Side pane with connection, telemetry (battery bar +
  full grid), flight, motion (WASD / Space-Shift / QE hold-to-fly with
  live velocity readout), and flips cards. Scrolling event log along the
  bottom.

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

### Phase B.5 — Perception / safety layer

The original Tello has **no forward distance sensor**, so obstacle
avoidance has to come from the camera. Two pieces, both local:

`tello/perception.py`:

1. **Optical-flow watchdog (always on)** — Farnebäck dense flow via
   `cv2.calcOpticalFlowFarneback` on the live video stream, ~10-15 Hz in
   a background thread. Computes the focus of expansion of the flow
   field; if vectors radiate outward strongly, the drone is rushing
   toward something. Raises a `proximity_alert` flag → dashboard
   auto-hovers via `drone.set_velocity(0, 0, 0, 0)`, agent is notified.
   ~15-30 ms per frame, no ML.
2. **MiDaS depth check (on demand)** —
   `check_path_clear(direction) -> {clear, closest_band, confidence}`.
   Loads MiDaS v2.1 Small ONNX once at startup via `cv2.dnn.readNet`.
   Samples a few frames over ~200 ms, averages the depth maps, inspects
   the center band of the predicted depth for "near" pixels. Returns a
   decision the agent acts on. ~60-100 ms per call.

The agent calls `check_path_clear` before any forward / lateral / up
move. The optical-flow watchdog runs unconditionally while the drone is
in flight and overrides the agent if it trips. Together with the
small-step movement pattern (≤ 50 cm per move), this bounds the worst
case to a low-velocity bump rather than a collision.

Goal: drone can be told to "explore" a cluttered room and reliably stops
short of furniture, walls, and people without operator intervention.
Bench-tested with hand-held obstacles before any autonomous flight.

### Phase C — Agent skeleton

`tello/agent.py`: OpenAI Agents SDK loop with tools:

- `take_off()`, `land()`, `emergency_stop()`, `move(direction, cm)`,
  `rotate(direction, degrees)` — wrap the existing `Drone` methods.
  Move tools **internally call `perception.check_path_clear(direction)`
  first** and refuse the move (returning a "path blocked" result the
  agent can reason about) if it isn't safe.
- `analyze_current_view()` — wraps `vision.analyze_frame()`
- `report_real_fire(description, severity)` /
  `report_false_alarm(reason)` — terminal tools, end the loop
- `return_and_land()` — terminal tool, safe abort

The optical-flow watchdog from Phase B.5 runs alongside the agent and
can interrupt any in-progress move (drone auto-hovers, agent gets a
proximity-alert event in its tool result stream).

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

### Phase E — Notification + dispatcher dashboard + polish

Two deliverables.

**1. `tello/notifier.py`** — structured incident events published over
WebSocket. A new banner zone in the operator console (above the video)
renders the outcome in the same flat dark / amber language as the rest
of the UI:

- **Real fire** → red-bordered banner: "FIRE DEPARTMENT NOTIFIED",
  severity (low / medium / high), one-line description, timestamp, the
  thumbnail of the frame the agent classified on, and the agent's
  reasoning excerpt.
- **False alarm** → amber-bordered banner: "ALARM CLEARED", the reason
  category (cooking, steam, sunlight, sensor fault, candle, other), the
  agent's full explanation, the same thumbnail + reasoning excerpt.

Both outcomes also append a permanent incident entry to the event log so
the operator can scroll the history. Twilio SMS as a stretch.

**2. Dispatcher dashboard at `/dashboard`** — a second HTML page served by
the same FastAPI app, framed as the **firefighter's real-time view of an
in-progress incident**. Same `/ws/telemetry` + agent + notification
events as the operator console, rendered for an audience instead of an
operator:

- Full-bleed live video as the hero
- Alarm-detected banner across the top when active
- Agent reasoning streaming in large readable type beside the video
- Vision-call thumbnails as a horizontal filmstrip with their JSON
  results
- Drone state strip (battery / altitude / flight time) — read-only, no
  control surface
- At verdict time the notification banner takes over half the screen
  for ~3 s, then settles into the corner so the post-mortem detail
  stays visible

Implemented as `tello/static/dashboard.html` + `dashboard.css` +
`dashboard.js` in the same visual language as the operator console.
Demo posture: project `/dashboard` on a TV or second screen while the
operator flies from `/` on the laptop.

## Demo flow

1. Tello on the floor, laptop on Tello WiFi + USB ethernet. **Operator
   console** open on the laptop, **`/dashboard`** projected on a second
   screen / TV as the firefighter view. Mission pill = idle.
2. Play a fire-alarm clip through a speaker.
3. Audio detector lights the alarm badge; mission pill flips to **armed**.
4. Agent kicks off; reasoning streams into a new agent card:
   *"Fire alarm detected. Initiating inspection. Examining current view…"*
5. Drone takes off (mission pill flips to **flight**), rotates and steps
   through a short inspection pattern. HUD insets update live.
6. Each vision call drops a thumbnail + structured JSON into the agent
   card. Perception layer's path-clear checks log alongside.
7. Agent reaches a verdict and calls the terminal tool:
   - **Real fire** → red "FIRE DEPARTMENT NOTIFIED" banner appears above
     the video with severity, description, and thumbnail. Event log
     gains a permanent INCIDENT entry.
   - **False alarm** → amber "ALARM CLEARED" banner with the reason
     category and the full agent explanation. Same permanent event-log
     entry.
8. Drone lands; mission pill returns to **idle**. Operator `Esc` was
   available throughout as the kill switch.

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
- **Original Tello has no forward distance sensor.** Obstacle avoidance
  is software-only (MiDaS depth check before moves + optical-flow
  watchdog during moves — see Phase B.5) and probabilistic. It reduces
  but does not eliminate the chance of bumping something. Always fly in
  a cleared room with no glass / curtains / people within ~2 m, at low
  altitude.
- No physical RC override on the original Tello — `Esc` from the
  dashboard is the only kill switch.
- Have 2-3 charged batteries on hand for the demo (Tello hover ~13 min).
