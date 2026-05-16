# FireDrone — Tello

Tello variant of FireDrone. A self-contained web dashboard for live control,
telemetry, and video, built as the foundation for the autonomous fire-response
agent.

This folder is an independent uv project. The original NimbusOS / BetaFPV
implementation under `../firedrone/` is preserved untouched.

## Hardware

- **DJI Tello** (original, not EDU). Speaks UDP SDK directly over its own WiFi AP.
- **Laptop** with WiFi (for the drone) and **USB ethernet** (for internet /
  OpenAI in later milestones).

## Networking

The original Tello broadcasts its own WiFi (`TELLO-XXXXXX`). Your laptop joins
it and reaches the drone at `192.168.10.1`. That WiFi has no internet.

Once we wire in OpenAI, we use a second NIC (USB ethernet) for internet. For
this milestone, only the Tello connection matters.

Smoke test before running anything: laptop joined to Tello WiFi, then

```bash
ping 192.168.10.1
```

If that fails, fix WiFi before continuing.

## Setup

```bash
cd tello
uv sync
```

## Milestone 1 — Web dashboard

Live video, telemetry, and full keyboard + on-screen control from a browser.

Run the dashboard:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> in a browser.

### Smoke scripts

Run these first to verify the SDK chain works before touching the web app.

```bash
uv run python scripts/smoke_telemetry.py   # 5s of telemetry, no flight
uv run python scripts/smoke_video.py       # OpenCV window with live feed
```

### Controls

| Key              | Action                          |
| ---------------- | ------------------------------- |
| `T`              | Takeoff                         |
| `L`              | Land                            |
| `Esc`            | **EMERGENCY** (cuts motors)     |
| `W` / `S`        | Forward / back (30 cm)          |
| `A` / `D`        | Left / right (30 cm)            |
| `Space` / `Ctrl` | Up / down (30 cm)               |
| `Q` / `E`        | Yaw left / right (30°)          |
| `1` / `2` / `3` / `4` | Flip forward / back / left / right |

On-screen buttons mirror every keyboard action.

### Safety

- WebSocket disconnect (tab closed, network drop) → auto-emergency (motors cut)
- Big red EMERGENCY button on the dashboard, always reachable
- Battery indicator turns red below 15% (manual landing still required)
- Always fly in a cleared room — no obstacle avoidance, no physical RC override
