"""Async fan-out event bus.

Used to broadcast structured events to multiple WebSocket clients (the
operator console and the dispatcher dashboard) without coupling
producers to consumers. The 5 Hz telemetry pump stays on its dedicated
``/ws/telemetry`` endpoint; everything else (vision results, audio
levels, agent reasoning, perception alerts, incidents) flows through
this bus and is served via ``/ws/events``.

Producers can be either:

* Async coroutines on the FastAPI event loop -> call ``bus.publish``.
* Plain threads (mic capture, agent worker, perception watchdog) ->
  call ``bus.publish_threadsafe``. The bus uses
  ``asyncio.run_coroutine_threadsafe`` so the publish actually happens
  on the loop thread without any explicit lock on the subscriber set.

Subscribers receive their own ``asyncio.Queue`` with a bounded buffer.
A slow subscriber drops events instead of blocking the producer or
blowing up memory — losing a couple of audio_level samples is far less
bad than wedging the agent loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("tello.events")

# Bounded so a stuck client can't grow memory without bound. 128 covers
# ~25 s of audio_level events at the 5 Hz cadence we publish them.
_SUB_QUEUE_MAX = 128


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUB_QUEUE_MAX)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    async def publish(self, event: dict[str, Any]) -> None:
        event.setdefault("ts", time.time())
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "subscriber queue full, dropping event %s", event.get("type")
                )

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Schedule a publish from a non-async thread."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)


# Process-wide singleton. Modules import ``bus`` and use it directly.
bus = EventBus()
