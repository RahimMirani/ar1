# FireDrone — Plan of Action

## What we're building

An autonomous fire-response drone agent. When a fire alarm sounds in the house, the drone arms, flies a quick inspection of the rooms via its camera, and decides whether it's a **real fire** or a **false alarm**. If real, it sends a notification (simulated fire department alert). If false, it logs the incident and lands. Optional stretch goal: drop a suppression pod on small fires.

The pitch is the **false-alarm filter**: most residential alarms are false, and dispatching a truck is expensive. An autonomous agent that verifies before escalating is the product.

## Hardware we have

- **BetaFPV Air75** — 21g indoor whoop with an analog C03 FPV camera. ELRS 2.4 GHz control, 5.8 GHz analog video.
- **DroneForge hardware** — USB dongle that plugs into the laptop. Receives the drone's 5.8 GHz video and transmits ELRS control to it. This is our bridge between laptop and drone.
- **RadioMaster Pocket** — ELRS radio controller. Manual override / safety.
- **Laptop** — runs everything (mic, AI, dashboard, drone control).

## Software stack

- **NimbusOS application** — runs in the background. It's a ZeroMQ server on localhost (pub `:7771`, sub `:7772`) that owns the USB connection to the DroneForge hardware. We do not use its UI. Keep it running, ignore it.
- **`nimbusos-sdk`** (Python package) — our interface to the drone. High-level API: arm/disarm, waypoint commands in meters, yaw turns, telemetry, state, camera frames.
- **Our custom Python app** — does everything else: mic input, audio classification, vision, agent reasoning, notifications, dashboard.

## What we are replacing

We're **not** using DroneForge desktop UI. We're **not** writing low-level CRSF / channel code. We use the SDK and build all our logic on top.

## Architecture

```
[Mic] ──► audio.py ──┐
                      ▼
[NimbusOS app]   agent.py ──► OpenAI (reasoning + tool calls)
   │ camera_frames  │  ▲
   │ state          │  │
   │ telemetry      │  │ vision (CLIP local + OpenAI vision)
   │                ▼  │
   │           drone.py ──► publish_arm_state / waypoint / yaw / land
   │                │
   └────────────────┘
            │
   ┌────────┴───────────┐
   ▼                    ▼
[Notifier]    [FastAPI dashboard on localhost]
```

## Tech choices

- Python 3.12, `uv` for env management
- `nimbusos-sdk` (pin version on day 1, never update)
- `sounddevice` for mic, YAMNet or spectral analysis for fire-alarm detection
- CLIP (`openai/clip-vit-base-patch32`) for fast on-frame fire/smoke classification
- **OpenAI API** for the agent loop with tool/function calling, and OpenAI vision on key frames
- FastAPI + vanilla HTML/JS/Tailwind (CDN) for the dashboard
- Notifier: simulated fire department alert on the dashboard (a fake outgoing message banner). Twilio SMS is optional if time permits.

## Plan of action

### Smoke tests first (everyone, do not skip)

1. `git init` new repo, `uv init`, `uv add nimbusos-sdk` (pinned)
2. NimbusOS app running, drone powered: run `print_telemetry.py` from the sandbox → expect telemetry output
3. Run `cam_test.py` from the sandbox → expect drone camera feed in a window
4. Run `getting_started.py` from the sandbox **with props off** → expect arm + small waypoint + land
5. Verify OpenAI API works with a one-shot test call

If any of these fail, fix before moving on. Nothing else matters.

### Tracks (parallel)

**Track 1 — `drone.py`**
Owns all drone movement. Wraps `NimbusClient` with semantic methods the agent will call: `arm()`, `disarm()`, `takeoff(altitude_m)`, `go_to(forward, right, altitude)`, `rotate(degrees)`, `hover()`, `land_and_disarm()`, `current_position()`, `battery_voltage()`. Includes safety: hard time cap on flight, auto-land on low battery. Each method tested in `tests/bench/` with props off before anything flies.

**Track 2 — `audio.py`**
Owns fire alarm detection. Reads live mic via `sounddevice.InputStream`, runs classification, emits a debounced `fire_alarm_detected` event (require N consecutive positive windows so random noise doesn't trigger it). Start with YAMNet (has a built-in smoke alarm class); fall back to spectral analysis tuned to ~3-4 kHz pulse pattern if YAMNet is painful. Tested against YouTube fire alarm clips played through a speaker.

**Track 3 — `vision.py`**
Owns "what is the drone seeing." Subscribes to `client.camera_frames()`, keeps the latest frame in memory, exposes two pipelines: a fast local CLIP classifier (labels like `["visible flames or fire", "thick smoke", "normal indoor scene"]`) running every frame, and a slower `analyze_frame_with_openai(jpeg_bytes)` that returns structured JSON for richer reasoning. The agent uses both. Tested against a folder of ~20 fire/non-fire images.

**Track 4 — `agent.py` + `server.py`**
Owns the decision-making and the dashboard. OpenAI agent loop with function calling. Tools exposed to the model: `start_inspection()`, `move_to_next_room()`, `analyze_current_view()`, `report_real_fire(description)`, `report_false_alarm(reason)`, `return_and_land()`. System prompt frames the agent as an autonomous fire safety operator. FastAPI + websocket pushes events to a single-page dashboard showing live camera, agent reasoning stream, audio meter, drone state, and action log.

**Stub everything first.** Each module exposes its public API with `pass` / fake returns from the start. Teammates import each other and build against stubs; real implementations land in parallel.

### Integration

Wire the tracks. Test the full flow with the drone bench-mounted (props off): play fire alarm → audio fires → agent starts → reads camera → makes decision → "lands."

Then test in flight in a small clear room.

### Rehearsal + polish

Run end-to-end ≥5 times. Demo script, backup batteries (4+ charged), kill-switch tested, dashboard looks clean.

## Demo flow

1. Drone on the floor, NimbusOS app running, our app running, dashboard on screen
2. Play fire alarm sound through a speaker
3. Audio detector lights up on dashboard → agent triggered
4. Drone arms, takes off, hovers
5. Agent reasoning streams to dashboard: "Fire alarm detected. Beginning inspection. Examining current view..."
6. Drone rotates / moves to inspect (a couple of waypoints in the room)
7. Agent decides: false alarm (no fire visible) or real fire (whatever we use to fake it)
8. If real → dashboard shows a simulated "Fire Department Notified" alert with severity + description; if false → log on dashboard
9. Drone lands and disarms

## Cut-scope priority (if behind)

Cut in this order:
1. Pod-dropping (already deprioritized)
2. Multi-room waypoint patrol → just hover and rotate to look around
3. Autonomous flight → fly via RadioMaster while the agent operates on the video feed (still impressive)

Do not cut: the dashboard, visible agent reasoning, false-vs-real decision, simulated notification.

## Safety rules

- Always bench test (props off) before any new flight code
- RadioMaster bound and tested as override before any autonomous flight
- Hard time cap on flight (auto-land after 90s) in `drone.py`
- Auto-land if battery voltage drops below threshold
- Demo room cleared of people during flight rehearsal