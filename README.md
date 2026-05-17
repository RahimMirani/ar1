# FireDrone

Autonomous fire-response drone agent. When a fire alarm sounds, the drone
arms, flies a quick inspection of the room via its camera, and decides
whether the alarm is **real** or **false** before notifying the (simulated)
fire department.

See [`plan.md`](plan.md) for the mission, architecture, and roadmap.

The live code is under [`tello/`](tello/). This document is the operator's
quick-start.

## Hardware

- **DJI Tello** (original, not EDU). UDP SDK over its own WiFi AP.
- **Laptop** — runs everything. WiFi NIC joins the Tello's network; USB
  ethernet provides internet for OpenAI calls (needed once Phase A lands;
  not required for the dashboard alone).

## Networking

The Tello broadcasts `TELLO-XXXXXX` (no internet) and sits at `192.168.10.1`.

Smoke test before running anything:

```bash
ping 192.168.10.1
```

If that fails, fix WiFi before continuing.

### Windows-specific gotcha

The Tello's WiFi must be set to the **Private** network profile, otherwise
inbound UDP 8890 / 11111 (state + video) are silently dropped even with
explicit firewall allow rules. Set it once per SSID in Windows Settings →
Network & internet → Wi-Fi → click the Tello network → change profile.

Recommended firewall rules (run as Administrator, once):

```powershell
New-NetFirewallRule -DisplayName "Tello SDK state" -Direction Inbound -Protocol UDP -LocalPort 8890 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "Tello SDK video" -Direction Inbound -Protocol UDP -LocalPort 11111 -Action Allow -Profile Any
```

## Setup

One time, from the repo root:

```bash
uv sync
```

That installs everything into `.venv/`. With uv you do **not** activate the
venv — `uv run <cmd>` handles that automatically.

## Smoke scripts

Run these first to confirm the SDK chain works before touching the dashboard.

```bash
uv run python tello/scripts/smoke_telemetry.py   # 5 s of telemetry, no flight
uv run python tello/scripts/smoke_video.py       # OpenCV window with live feed
```

## Run the dashboard

```bash
uv run --directory tello uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in a browser, click **Connect**, fly.

## Controls

Live motion is **hold-to-fly**: press and hold a key to fly continuously,
release to stop. Combinations work naturally (e.g. `W + D` flies diagonally
forward-right).

| Key                       | Action                              |
| ------------------------- | ----------------------------------- |
| `T`                       | Takeoff                             |
| `L`                       | Land                                |
| `Esc`                     | **EMERGENCY** (cuts motors)         |
| `W` / `S`                 | Forward / back (hold)               |
| `A` / `D`                 | Left / right (hold)                 |
| `Space` / `Shift`         | Up / down (hold)                    |
| `Q` / `E`                 | Yaw left / right (hold)             |
| `1` / `2` / `3` / `4`     | Flip forward / back / left / right  |

On-screen motion buttons mirror the keyboard with click-and-hold.

## Safety

- WebSocket disconnect (tab closed, network drop) → auto-emergency (motors cut)
- Big red EMERGENCY button on the dashboard, always reachable
- Battery indicator turns red below 15% (manual landing still required)
- Always fly in a cleared room — no obstacle avoidance, no physical RC override
