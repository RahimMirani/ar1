# FireDrone

Autonomous fire-response drone agent. When a fire alarm is detected, an OpenAI
agent flies the BetaFPV Air75 around the room via the DroneForge dongle, checks
camera frames against vision models, and decides whether the alarm is real
before notifying the (simulated) fire department.

See [plan.md](plan.md) for the full project plan.

## Milestone 1: keyboard teleop

Goal: control the drone live from the laptop using the keyboard, on top of a
`Drone` wrapper that the rest of the codebase will share.

### Prereqs

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/) installed
- NimbusOS desktop app running and connected to the DroneForge dongle
- Drone powered on, propellers **OFF** for first bench tests

### Setup

```bash
uv sync
```

### Smoke test the SDK connection

With NimbusOS running and the drone on, this should print live telemetry for
~5 seconds:

```bash
uv run firedrone-telemetry
```

If nothing prints, fix NimbusOS / the dongle before going further.

### Teleop

```bash
uv run firedrone-teleop
```

Keys:

| Key       | Action                              |
| --------- | ----------------------------------- |
| `T`       | Arm + takeoff to 1.0 m              |
| `L`       | Land + disarm                       |
| `Esc`     | Emergency land + exit               |
| `W` / `S` | Step forward / back (0.2 m)         |
| `A` / `D` | Step left / right (0.2 m)           |
| `Space` / `Ctrl` | Step up / down (0.2 m)       |
| `Q` / `E` | Yaw left / right (~11.5 deg)        |
| `H`       | Hover (cancel current motion)       |

### Safety

- Hard 90-second flight cap (`Drone` watchdog auto-lands)
- Auto-land below 3.4 V battery
- Position commands are clamped to a small indoor envelope
- Always do first runs with props off and the RadioMaster bound as override
