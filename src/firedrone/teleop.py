"""Keyboard teleop for the FireDrone.

Run with:

    uv run firedrone-teleop

Keys
----
T          arm + takeoff to 1.0 m
L          land + disarm
Esc        emergency land + exit
W / S      forward / back (0.2 m per tick)
A / D      left  / right  (0.2 m per tick)
Space      up             (0.2 m per tick)
X          down           (0.2 m per tick)
Q / E      yaw left / right (~11.5 deg per press)
H          hover (cancel motion)

`Drone` owns all SDK calls and safety logic; this module only translates
keypresses into method calls and renders the HUD.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from pynput import keyboard
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from firedrone.drone import Drone, DroneSnapshot

TICK_HZ = 10.0
STEP_M = 0.2
YAW_STEP_RAD = 0.2  # ~11.5 degrees per press

HELD_KEYS = {"w", "s", "a", "d", "x"}  # space handled separately
HELD_KEYS_WITH_SPACE = HELD_KEYS | {"space"}
EDGE_KEYS = {"q", "e", "t", "l", "h", "esc"}


def _normalize(key: keyboard.Key | keyboard.KeyCode | None) -> Optional[str]:
    """Return a lowercase string id for a pynput key, or None if unknown."""
    if key is None:
        return None
    if isinstance(key, keyboard.KeyCode):
        if key.char is None:
            return None
        return key.char.lower()
    name = getattr(key, "name", None)
    if name is None:
        return None
    return name.lower()


class TeleopController:
    """Owns the keyboard listener, the tick loop, and the rich HUD."""

    def __init__(self, drone: Drone) -> None:
        self._drone = drone
        self._console = Console()
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._edge_queue: list[str] = []
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Keyboard listener callbacks (run on pynput's thread)
    # ------------------------------------------------------------------ #

    def _on_press(self, key: object) -> None:
        name = _normalize(key)  # type: ignore[arg-type]
        if name is None:
            return
        with self._lock:
            if name in HELD_KEYS_WITH_SPACE:
                self._pressed.add(name)
            if name in EDGE_KEYS and name not in self._pressed:
                self._edge_queue.append(name)
                self._pressed.add(name)

    def _on_release(self, key: object) -> None:
        name = _normalize(key)  # type: ignore[arg-type]
        if name is None:
            return
        with self._lock:
            self._pressed.discard(name)

    # ------------------------------------------------------------------ #
    # Tick handlers
    # ------------------------------------------------------------------ #

    def _drain_edges(self) -> list[str]:
        with self._lock:
            edges = self._edge_queue
            self._edge_queue = []
        return edges

    def _held_set(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def _apply_held(self, held: set[str]) -> None:
        d_forward = 0.0
        d_right = 0.0
        d_down = 0.0
        if "w" in held:
            d_forward += STEP_M
        if "s" in held:
            d_forward -= STEP_M
        if "d" in held:
            d_right += STEP_M
        if "a" in held:
            d_right -= STEP_M
        if "space" in held:
            d_down -= STEP_M  # NED: negative down == up
        if "x" in held:
            d_down += STEP_M
        if d_forward == 0.0 and d_right == 0.0 and d_down == 0.0:
            return
        try:
            self._drone.nudge(d_forward=d_forward, d_right=d_right, d_down=d_down)
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[red]nudge failed:[/red] {exc}")

    def _apply_edge(self, key: str) -> None:
        try:
            if key == "t":
                self._drone.takeoff()
            elif key == "l":
                self._drone.land_and_disarm()
            elif key == "h":
                self._drone.hover()
            elif key == "q":
                self._drone.rotate(-YAW_STEP_RAD)
            elif key == "e":
                self._drone.rotate(YAW_STEP_RAD)
            elif key == "esc":
                self._drone.emergency_land()
                self._stop.set()
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[red]{key} command failed:[/red] {exc}")

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #

    def _render(self, snap: DroneSnapshot, held: set[str]) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        if snap.armed and snap.flying:
            status_text = Text(snap.last_status, style="bold green")
        elif snap.armed:
            status_text = Text(snap.last_status, style="bold yellow")
        else:
            status_text = Text(snap.last_status, style="bold red")
        table.add_row("status", status_text)

        battery = (
            f"{snap.battery_volts:.2f} V" if snap.battery_volts is not None else "--"
        )
        battery_style = "green"
        if snap.battery_volts is not None and snap.battery_volts < 3.6:
            battery_style = "yellow"
        if snap.battery_volts is not None and snap.battery_volts < 3.4:
            battery_style = "bold red"
        table.add_row("battery", Text(battery, style=battery_style))

        yaw = f"{snap.yaw_deg:6.1f} deg" if snap.yaw_deg is not None else "--"
        table.add_row("yaw", yaw)

        if snap.x_m is not None:
            pos = f"x={snap.x_m:6.2f}  y={snap.y_m:6.2f}  z={snap.z_m:6.2f}"
        else:
            pos = "--"
        table.add_row("position", pos)

        target = (
            f"f={snap.target_forward_m:6.2f}  "
            f"r={snap.target_right_m:6.2f}  "
            f"down={snap.target_down_m:6.2f}"
        )
        table.add_row("target", target)

        table.add_row("flight", f"{snap.flight_seconds:5.1f} s")
        table.add_row("keys", " ".join(sorted(held)) if held else "-")

        controls = Text.from_markup(
            "[b]T[/b] takeoff   [b]L[/b] land   [b]H[/b] hover   "
            "[b]Esc[/b] EMERGENCY\n"
            "[b]W/A/S/D[/b] move   [b]Space/X[/b] up/down   [b]Q/E[/b] yaw"
        )

        body = Table.grid()
        body.add_row(table)
        body.add_row(Text(""))
        body.add_row(controls)
        return Panel(body, title="FireDrone teleop", border_style="cyan")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        try:
            with Live(
                self._render(self._drone.snapshot(), set()),
                console=self._console,
                refresh_per_second=TICK_HZ,
                screen=False,
            ) as live:
                tick = 1.0 / TICK_HZ
                while not self._stop.is_set():
                    started = time.monotonic()
                    for key in self._drain_edges():
                        self._apply_edge(key)
                        if self._stop.is_set():
                            break
                    held = self._held_set()
                    if not self._stop.is_set():
                        self._apply_held(held)
                    live.update(self._render(self._drone.snapshot(), held))
                    elapsed = time.monotonic() - started
                    if elapsed < tick:
                        time.sleep(tick - elapsed)
        finally:
            listener.stop()


def main() -> None:
    with Drone() as drone:
        TeleopController(drone).run()


if __name__ == "__main__":
    main()
