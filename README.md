# AR1 (Aerial Rover 1)

<table>
  <tr>
    <td width="50%"><img alt="Operator console" src="https://github.com/user-attachments/assets/b7a96d55-80d6-4267-a5ae-0c2e528e931f" /></td>
    <td width="50%"><img alt="Dispatcher dashboard" src="https://github.com/user-attachments/assets/fceee167-7213-49ba-8d5c-b3413cb5f397" /></td>
  </tr>
</table>

> An autonomous fire-response drone. The moment a smoke alarm sounds it
> takes off, sweeps the room, and gives the fire department a verdict
> with evidence, before the truck rolls.

## Use Case

When a smoke alarm goes off, the fire department has two bad options:
roll a truck blind, or wait for human eyes on the scene. Most
residential calls turn out to be false, and waiting on humans is slow
when they aren't. **AR1 is the eyes on the scene.** The instant an
alarm trips, a DJI Tello takes off, patrols the room, and streams a
live verdict (with reasoning and the deciding camera frames) to a
dispatcher dashboard.

- **Real fire** → red *FIRE DEPARTMENT NOTIFIED* banner with severity,
  description, and thumbnails. A webhook posts the incident to the
  (simulated) fire department so the crew already knows what they're
  walking into.
- **False alarm** → amber *ALARM CLEARED* banner with the agent's full
  explanation (*"candle in the living room, no smoke, no combustion
  signature"*) and the supporting thumbnails. Just as detailed as the
  real-fire payload, because it's the audit trail.

Then the drone lands.

## How it works

```
[Mic] ──▶ FFT alarm detector ──┐
                               ▼
                         Agent loop
                          (gpt-4o)
                               │
                  ┌────────────┼────────────┐
            vision             │           mapping
       (gpt-4o-mini +          │       dead-reckon +
        Pydantic schema)       │       occupancy grid
                               │
                         perception
                       MiDaS depth +
                     optical-flow watchdog
                               │
                               ▼
                          DJI Tello
                               │
                               ▼
                    FastAPI + WebSocket
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
             /  operator              /dashboard
             (the pilot)              (the firefighters)
```

1. **Listen.** A pure-DSP detector runs an FFT over the laptop mic and
   matches the UL-217 **T3 pulse cadence** (three half-second beeps in
   the 3 kHz band), so TV sibilants, microwave beeps, and speech don't
   trip it.
2. **Reason.** The alarm transition fires an **agent loop** running on
   `gpt-4o`, with the drone, vision, mapping and perception modules
   registered as callable tools. The agent gets a 240 s budget, a
   36-turn cap, and one terminal tool: `report_finding(verdict, reasons)`.
3. **See.** Each `analyze_view()` call sends the current frame to
   `gpt-4o-mini` with a **Pydantic-constrained JSON schema**, returning
   `fire_visible / smoke_visible / severity / confidence / reasons`.
   Every captured frame is logged to the dispatcher dashboard as
   evidence.
4. **Avoid.** The Tello has no forward distance sensor, so obstacle
   avoidance is software-only:
   - **Proactive.** Before every lateral move, **MiDaS v2.1 Small**
     (21 MB ONNX, run via `cv2.dnn`; no PyTorch, no transformers,
     ~150 ms on CPU) inspects the center patch of the predicted depth
     map; the move is shortened or refused if anything's in the way.
   - **Reactive.** A **Farnebäck dense optical-flow** watchdog runs at
     10 Hz. If the flow field radiates outward (focus-of-expansion =
     something rushing the camera), the drone hovers immediately.
5. **Remember.** A dead-reckoning mapper integrates `speed_x/y` + yaw
   at 10 Hz with an IMU-blended complementary filter and a *belly-cam
   lockout detector*: if accel says we're moving but optical flow reads
   zero, the position integral is lying, the pose is flagged
   low-confidence, and obstacle stamping pauses. The live trajectory
   and MiDaS-stamped occupancy grid are streamed as `/map.mjpg`.
6. **Decide.** The agent must visit at least four distinct positions
   before it's allowed to clear an alarm. Real-fire verdicts require
   two independent visual cues from different angles. The verdict
   streams to the dispatcher with reasoning and thumbnails, and a
   webhook POSTs the incident on real fires.

## The two browser views

- **`/`** is the operator console. Live MJPEG with glass HUD insets,
  hold-to-fly WASD/Space/Shift/QE, telemetry, depth-view toggle,
  live map toggle, audio meter, event log, big red emergency button.
  The pilot's seat.
- **`/dashboard`** is the dispatcher view. Same WebSocket event stream
  but read-only, laid out for an audience: full-bleed video, alarm
  banner, agent reasoning streaming in large type, vision thumbnails
  as a filmstrip, and a verdict banner that takes over the screen at
  decision time. Project this on a TV; the operator flies from `/`.

## Tech stack

| Layer            | Choice                                          | Why                                      |
| ---------------- | ----------------------------------------------- | ---------------------------------------- |
| Drone SDK        | `djitellopy` over UDP                           | Original Tello, no station mode          |
| Server           | FastAPI + uvicorn + WebSockets                  | One process, two browser views           |
| Agent loop       | `gpt-4o` with tool-calling                      | Reasoning + structured tool use          |
| Vision           | OpenAI `gpt-4o-mini` + Pydantic schema          | Cheap, fast, structured output           |
| Depth            | MiDaS v2.1 Small ONNX via `cv2.dnn`             | ~21 MB, no torch, CPU-only               |
| Motion watchdog  | Farnebäck dense optical flow (OpenCV)           | ~15 ms / frame, no ML                    |
| Pose & map       | Dead-reckon + IMU complementary filter          | Honest odometry, ~30 to 60 cm/min drift  |
| Audio            | `sounddevice` + numpy FFT                       | Pure DSP T3-cadence detector             |
| Frontend         | Vanilla HTML / CSS / JS, MJPEG                  | No build step, two pages                 |
| Env management   | Python 3.12, `uv`, single `pyproject.toml`      | Reproducible, fast                       |

## Safety: what's actually load-bearing

The Tello is fragile and the link is unreliable. Four things keep this
honest:

- **`Esc` cuts motors.** Always, from any state. Same as the dashboard's
  big red button.
- **WebSocket disconnect cuts motors.** Browser tab closes → drone falls
  a few centimetres instead of drifting blind.
- **Hard 240 s flight cap** inside the agent loop. Timeout = forced
  land + a synthesised `unknown` verdict so the dispatcher still gets
  an event.
- **Soft geofence** monitors state-packet loss, RTT, and video-decode
  errors. On degradation it auto-hovers, *once* per tier entry, so a
  failing link can't make the drone permanently un-flyable.

The depth check, watchdog, and pose-confidence flag don't eliminate the
chance of a bump; they bound the worst case to a low-velocity nudge in
a cleared room.

## Quick start

```bash
uv sync                                                       # one-time install
ping 192.168.10.1                                             # laptop must be on TELLO-XXXXXX WiFi
uv run --directory tello uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, click **Connect**, fly. Open
<http://127.0.0.1:8000/dashboard> on a second screen for the dispatcher
view.

Set `OPENAI_API_KEY` in `tello/.env` to enable the agent and vision
tools. Without it everything else (manual flight, depth view, map,
audio detector) still works.

### Networking

The Tello has its own WiFi AP with no internet, so the laptop needs two
NICs: WiFi joined to the drone, USB ethernet for OpenAI calls. Windows
routes by destination IP automatically.

One Windows gotcha: the Tello SSID must be set to the **Private**
profile, otherwise inbound UDP 8890 / 11111 are silently dropped even
with explicit firewall allow rules. One-time PowerShell as admin:

```powershell
New-NetFirewallRule -DisplayName "Tello SDK state" -Direction Inbound -Protocol UDP -LocalPort 8890  -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "Tello SDK video" -Direction Inbound -Protocol UDP -LocalPort 11111 -Action Allow -Profile Any
```

## Controls

Live motion is **hold-to-fly**: press and hold a key to fly continuously,
release to stop. Combinations work naturally (`W + D` = forward-right).

| Key                 | Action                              |
| ------------------- | ----------------------------------- |
| `T` / `L`           | Takeoff / Land                      |
| `Esc`               | **EMERGENCY** (cuts motors)         |
| `W` `A` `S` `D`     | Forward / left / back / right (hold)|
| `Space` / `Shift`   | Up / down (hold)                    |
| `Q` / `E`           | Yaw left / right (hold)             |
| `1` `2` `3` `4`     | Flip forward / back / left / right  |

## Project layout

- [`tello/drone.py`](tello/drone.py): the only module that talks to
  `djitellopy`. Owns connection, telemetry, RC velocity, link-safety
  patches.
- [`tello/main.py`](tello/main.py): FastAPI server, WebSockets, MJPEG.
- [`tello/agent.py`](tello/agent.py): the agent loop and its tool
  definitions.
- [`tello/vision.py`](tello/vision.py): `analyze_frame()` with the
  Pydantic verdict schema.
- [`tello/perception.py`](tello/perception.py): MiDaS depth check +
  Farnebäck optical-flow watchdog.
- [`tello/audio.py`](tello/audio.py): T3-cadence FFT alarm detector.
- [`tello/mapping.py`](tello/mapping.py): dead-reckoning pose +
  occupancy grid + IMU lockout detector.
- [`tello/depth_stream.py`](tello/depth_stream.py): live MiDaS
  visualisation stream.
- [`tello/notifier.py`](tello/notifier.py): incident events +
  optional dispatch webhook.
- [`tello/static/`](tello/static/): operator & dispatcher consoles
  (vanilla HTML/CSS/JS).

See [`plan.md`](plan.md) for the full architecture write-up and
[`AGENTS.md`](AGENTS.md) for the link-safety invariants every edit
must respect.
