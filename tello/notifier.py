"""Incident notifier — turns agent findings into incident events.

The agent ends every mission with one ``agent_finding`` event on the
bus. This module consumes those events and:

1. Bundles the verdict with the evidence trail (vision frames the
   agent captured during the run) and the mission timing into a single
   :class:`Incident` payload.
2. Publishes an ``incident`` event on the bus so the operator console
   and the dispatcher dashboard can render it.
3. Optionally POSTs the incident JSON to ``FIREDRONE_NOTIFY_WEBHOOK``
   (simulated fire-department endpoint) for real-fire verdicts. False
   alarms are logged but not dispatched — Phase E spec says the
   notification must explain *why* a verdict was reached, regardless of
   verdict, so the dispatcher dashboard always renders the full payload
   even when no webhook is fired.

The latest incident is kept in memory (single-mission demo) and served
via ``GET /api/incidents/latest`` so a dispatcher dashboard loaded
mid-incident immediately renders the current state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

import httpx

from events import bus

logger = logging.getLogger("tello.notifier")

WEBHOOK_URL = (os.getenv("FIREDRONE_NOTIFY_WEBHOOK") or "").strip() or None
WEBHOOK_TIMEOUT_SEC = 5.0

# Single in-memory cache. Single dashboard, single mission at a time.
_latest_incident: dict[str, Any] | None = None
# Evidence we accumulate while the agent runs — reset on every new mission.
_evidence_buffer: list[dict[str, Any]] = []
# Timeline of significant agent steps to surface to the dispatcher.
_timeline_buffer: list[dict[str, Any]] = []


def latest_incident() -> dict[str, Any] | None:
    """Return the most recent incident dict, or ``None`` if there hasn't been one."""
    return _latest_incident


def _title_for(verdict: str) -> str:
    return {
        "real_fire":   "Fire confirmed — dispatching fire department",
        "false_alarm": "False alarm — logged, no dispatch",
        "unknown":     "Inconclusive verdict — manual review required",
    }.get(verdict, verdict)


async def run_notifier_loop() -> None:
    """Long-running task: subscribe to the bus, emit incidents on findings.

    Spawned once from ``main.py``'s lifespan. Cancellation is handled by
    catching ``CancelledError`` and unsubscribing cleanly.
    """
    q = bus.subscribe()
    logger.info("notifier loop started (webhook=%s)", WEBHOOK_URL or "disabled")
    try:
        while True:
            ev = await q.get()
            t = ev.get("type")
            if t == "agent_state":
                state = ev.get("state")
                if state == "starting":
                    _evidence_buffer.clear()
                    _timeline_buffer.clear()
                _timeline_buffer.append(
                    {"kind": "state", "state": state, "ts": ev.get("ts")}
                )
            elif t == "vision_result" and ev.get("source") == "agent":
                _evidence_buffer.append(
                    {
                        "ts":            ev.get("ts"),
                        "severity":      ev.get("severity"),
                        "fire_visible":  ev.get("fire_visible"),
                        "smoke_visible": ev.get("smoke_visible"),
                        "confidence":    ev.get("confidence"),
                        "description":   ev.get("description"),
                        "reasons":       ev.get("reasons", []),
                        "thumbnail_b64": ev.get("thumbnail_b64"),
                        "model":         ev.get("model"),
                        "source":        ev.get("source"),
                        "latency_ms":    ev.get("latency_ms"),
                    }
                )
            elif t == "agent_tool_call":
                _timeline_buffer.append(
                    {
                        "kind": "tool_call",
                        "tool": ev.get("tool"),
                        "args": ev.get("args"),
                        "ts":   ev.get("ts"),
                    }
                )
            elif t == "agent_finding":
                await _emit_incident(ev)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("notifier loop crashed: %s", exc)
    finally:
        bus.unsubscribe(q)


async def _emit_incident(finding: dict[str, Any]) -> None:
    global _latest_incident
    verdict = str(finding.get("verdict", "unknown"))
    summary = str(finding.get("summary") or "")
    reasons = list(finding.get("reasons") or [])

    # Dispatch policy: only real fires "page" the fire department.
    notified = verdict == "real_fire"

    incident = {
        "type": "incident",
        "incident_id": uuid.uuid4().hex[:10],
        "mission_id":  finding.get("mission_id"),
        "verdict":     verdict,
        "title":       _title_for(verdict),
        "summary":     summary,
        "reasons":     reasons,
        "evidence":    list(_evidence_buffer),
        "timeline":    list(_timeline_buffer),
        "notified_dept": notified,
        "webhook_url": WEBHOOK_URL,
        "ts": time.time(),
        "synthesised": bool(finding.get("synthesised", False)),
    }
    _latest_incident = incident
    await bus.publish(incident)
    logger.info(
        "incident %s verdict=%s notified=%s evidence=%d",
        incident["incident_id"], verdict, notified, len(incident["evidence"]),
    )

    if WEBHOOK_URL and notified:
        await _post_webhook(incident)


async def _post_webhook(incident: dict[str, Any]) -> None:
    """Best-effort POST to the configured webhook. Failures are logged
    but don't propagate — the dispatcher dashboard is the source of
    truth for the demo.

    We strip evidence thumbnails before posting so the payload stays
    light (the dispatcher dashboard already has the bus stream for those).
    """
    light = dict(incident)
    light["evidence"] = [
        {k: v for k, v in entry.items() if k != "thumbnail_b64"}
        for entry in incident.get("evidence", [])
    ]
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SEC) as client:
            resp = await client.post(WEBHOOK_URL, json=light)
        logger.info("webhook %s -> %s", WEBHOOK_URL, resp.status_code)
        await bus.publish(
            {
                "type": "webhook_delivery",
                "incident_id": incident["incident_id"],
                "url": WEBHOOK_URL,
                "status": resp.status_code,
            }
        )
    except Exception as exc:
        logger.warning("webhook post failed: %s", exc)
        await bus.publish(
            {
                "type": "webhook_delivery",
                "incident_id": incident["incident_id"],
                "url": WEBHOOK_URL,
                "status": None,
                "error": str(exc),
            }
        )
