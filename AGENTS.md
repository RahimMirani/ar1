# Agent guide — firedrone-tello

This file is the orientation note for any agent (AI or human) editing the
codebase. Read it before touching `tello/drone.py`. For project-level
context (what we're building, why, milestones) read `plan.md`.

## Project shape

- **`tello/drone.py`** — the only module that talks to `djitellopy.Tello`
 directly. Owns connection, telemetry, RC velocity, fence, monkey-patches.
- **`tello/main.py`** — FastAPI server. Websocket telemetry + control,
 MJPEG video, `/api/connect`.
- **`tello/perception.py`** — MiDaS path-check + Farnebäck flow watchdog.
 Pure consumer of `Drone` accessors.
- **`tello/depth_stream.py`** — live MiDaS visualisation tile rendered
 at ~3 Hz and served on `/depth.mjpg`. Pure consumer.
- **`tello/mapping.py`** — 2D dead-reckoned pose + MiDaS occupancy grid.
 Pure consumer. Two daemon threads (pose @ 10 Hz, render @ 2 Hz).
 Auto-resets on the takeoff transition it observes via `get_flying()`.
 Served as a JPEG stream on `/map.mjpg` + JSON snapshot on
 `/api/map/snapshot`. The agent reads from it via the `get_pose` and
 `get_map_summary` tools.
- **`tello/static/`** — operator console (vanilla HTML/CSS/JS, no build).
- **`tello/scripts/`** — standalone smoke tests. `smoke_video.py` and
 `smoke_telemetry.py` need a drone; `smoke_link.py` and
 `smoke_mapping.py` do not.

## Run

```bash
uv run --directory tello uvicorn main:app --host 127.0.0.1 --port 8000
```

Dashboard at `http://127.0.0.1:8000`. The laptop's WiFi NIC must be joined
to the Tello's `TELLO-XXXXXX` access point.

## Link safety — the must-not-break list

The dashboard's responsiveness and the soft geofence depend on a tight
set of invariants. They were each added in response to a regression we
hit in real flight testing. Detail and reasoning live in the module
docstring of `tello/drone.py` under "LINK SAFETY INVARIANTS"; the
summary below is what every edit must respect.

1. **Three monkey-patches must apply at import.** Set on `Tello` and
   `BackgroundFrameRead` from `djitellopy`:
   - `Tello.send_command_with_return` is wrapped with
     `_command_send_lock` to serialize the SDK command channel.
   - `Tello.parse_state` is wrapped to count incoming state packets.
   - `BackgroundFrameRead.update_frame` is replaced with a resilient
     decoder that survives corrupt H.264 packets and counts decode errors.

   The patches each set a `_*_patched` flag and an assertion at the
   bottom of the patch block raises `RuntimeError` at import if any did
   not apply. Don't suppress that assertion.

2. **RC velocity stays on the fire-and-forget SDK path.**
   `Drone.set_velocity` calls `send_rc_control` (no ack expected). Do
   not route teleop through `send_control_command` or any `query_*`
   method — those wait for an ack, go through the serialization lock,
   and add the wifi-poll RTT to every keypress.

3. **Fence tuning is a coupled system.** `_link_history` maxlen,
   `_LINK_WARMUP_SEC`, `_FENCE_DEBOUNCE_SEC`, and the per-tier
   thresholds were chosen together to avoid false HOVER triggers on a
   healthy link. Don't change one without re-running `smoke_link.py`.

4. **`_link_history.clear()` on takeoff is load-bearing.** The Tello
   pauses state broadcasts during motor spin-up; without the clear that
   pause reads as ~20% loss for the first 5 s of flight and trips HOVER
   before the drone is airborne.

5. **The fence's HOVER override is one-shot per tier entry.** Fired in
   `_apply_fence_transition` exactly once when HOVER is first committed.
   Do not call `set_velocity` on every tick HOVER is active — that
   would make the drone permanently un-flyable until the link recovers.

## Threading model

Three background threads, started in `Drone._start_background_threads`:

| Thread | Rate | Touches the response queue? |
|---|---|---|
| `tello-telemetry` | 5 Hz | No — reads state cache only |
| `tello-rc`        | 20 Hz | No — uses `send_command_without_return` |
| `tello-wifi`      | 1 Hz | Yes (via the lock) |

The dashboard's operator commands (takeoff, land, flip, …) also touch
the response queue via the lock. RC velocity does not — that's
intentional, see §2.

Four optional consumer threads run alongside, each owned by its own
module. They start on operator toggle (or, for the watchdog, on takeoff)
and shut down with the FastAPI lifespan. None of them touch djitellopy
directly — they pull data through `Drone.snapshot()` and
`Drone.get_frame()`. New consumer modules should follow the same shape.

| Thread | Owner | Rate | Purpose |
|---|---|---|---|
| `tello-perception` | `OpticalFlowWatchdog` | 10 Hz | Farnebäck flow; halts on spike |
| `tello-depth-stream` | `DepthStream` | 3 Hz | MiDaS render → `/depth.mjpg` |
| `tello-mapper-pose` | `Mapper` | 10 Hz | Pose integrator + IMU lockout detector |
| `tello-mapper-render` | `Mapper` | 2 Hz | MiDaS occupancy stamping + map JPEG |

`Drone.link_diagnostics()` reports the patch flags + thread liveness as
booleans; `/api/connect` includes it in the response, and the dashboard
logs any False entries to the event log. Use it as your first
sanity-check after any edit.

## Verifying changes

Run before pushing edits that touch `tello/drone.py`:

```bash
uv run python tello/scripts/smoke_link.py
```

And before pushing edits that touch `tello/mapping.py`:

```bash
uv run python tello/scripts/smoke_mapping.py
```

Neither needs a drone. Together they take under 30 seconds and cover
every invariant either file is supposed to keep.

If the smoke test passes but the dashboard still misbehaves, check the
event log on connect — `link safety degraded: …` means a monkey-patch
didn't apply or a thread didn't start, which the smoke test should have
caught (please add a case if not).

## Commit hygiene

- Keep commits atomic — one logical change per commit.
- Commit messages start with a Conventional Commits prefix
  (`feat(drone): …`, `fix(drone): …`, `docs: …`, `chore(scripts): …`).
- Do not edit `plan.md` as part of a code change unless the user asks
  for it explicitly.
- Do not start long-running servers (uvicorn etc.) from inside the
  agent — the operator runs the dashboard themselves.

## Mapping — what `tello/mapping.py` promises (and doesn't)

The mapping layer is honest dead-reckoning + MiDaS, not real SLAM. A
few things are worth knowing before editing it or building on top:

1. **It's a pure consumer.** Reads telemetry, frame, and `flying` via
   the `Drone` accessors passed into the constructor. Never imports
   `djitellopy`. New mapping-adjacent code should keep that contract.
2. **Pose is integrated from `speed_x/y` + `yaw_deg` with an
   IMU-blended complementary filter.** Realistic drift is ~30-60 cm
   per minute on a textured indoor floor; expect more on featureless
   surfaces. There is no loop closure — the map is a per-mission
   scratchpad, not a persistent world model.
3. **Belly-cam lockout flips confidence to `low` and freezes the
   position integral.** This is what stops a drift episode from being
   silently baked into the map. The detector is in `_integrate_pose`;
   thresholds (`LOCKOUT_VEL_CMPS`, `LOCKOUT_ACCEL_MG`,
   `LOCKOUT_CONSECUTIVE_TICKS`) are coupled — tune them together and
   re-run `smoke_mapping.py §4`.
4. **The takeoff transition resets the map.** The pose loop watches
   `flying` going False → True and calls `reset()`. Do not couple a
   reset call into `Drone.takeoff()` instead — that would re-introduce
   a path from the safety module to the consumer module and break the
   AGENTS contract.
5. **Coordinate convention is +x = takeoff-forward, +y =
   operator-right.** Tello yaw is CW-positive, and the rotation in
   `_integrate_pose` is the textbook 2D matrix. The earlier draft
   used a non-standard convention; `smoke_mapping.py §2` is the
   regression that catches it.
6. **MiDaS distances are non-metric.** Obstacle cells are stamped as
   a coarse fan (`OBS_STAMP_NEAR_M..OBS_STAMP_FAR_M`) at the camera
   bearing, not at a precise range. Treat the map as "there is
   something in that direction" rather than "the wall is at 1.42 m".

If you add a new SDK accessor the mapper should read, plumb it through
`drone._read_telemetry()` first so the rest of the project sees it
too. Don't reach around the `Drone` class.

## Pointers for future work

The next phases described in `plan.md` (Vision spine, Audio trigger,
Perception layer, Agent skeleton) will all eventually call into the
`Drone` class. None of them should bypass it to talk to djitellopy
directly. If you need a new SDK capability, add a method to `Drone`
that wraps it correctly (acquire the existing locks, update
`_last_status`, route through `send_command_with_return` so the lock
applies).
