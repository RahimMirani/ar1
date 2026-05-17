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

Live motion is **hold-to-fly**: press and hold a key to fly continuously in
that direction, release to stop. Hold combinations work naturally
(e.g. `↑ + →` flies diagonally forward-right).

| Key                       | Action                              |
| ------------------------- | ----------------------------------- |
| `T`                       | Takeoff                             |
| `L`                       | Land                                |
| `Esc`                     | **EMERGENCY** (cuts motors)         |
| `↑` / `↓`                 | Forward / back (hold)               |
| `←` / `→`                 | Left / right (hold)                 |
| `Space` / `Shift`         | Up / down (hold)                    |
| `Q` / `E`                 | Yaw left / right (hold)             |
| `1` / `2` / `3` / `4`     | Flip forward / back / left / right  |

On-screen motion buttons mirror the keyboard with click-and-hold.

### Safety

- WebSocket disconnect (tab closed, network drop) → auto-emergency (motors cut)
- Big red EMERGENCY button on the dashboard, always reachable
- Battery indicator turns red below 15% (manual landing still required)
- Always fly in a cleared room — no obstacle avoidance, no physical RC override
