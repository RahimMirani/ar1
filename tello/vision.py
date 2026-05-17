"""OpenAI vision wrapper for fire/smoke detection from camera frames.

The dashboard's "Analyze current view" button and the agent's
``analyze_view`` tool both go through :func:`analyze_frame`. We use
``gpt-4o-mini`` — a strong-enough multimodal model that runs in well
under a second per frame on the network paths we've tested, at ~1/10th
the cost of full ``gpt-4o``. The agent reasons over a Pydantic
``FireDetection`` schema rather than free-form prose so the verdict is
deterministic-shape downstream.

The thumbnail returned in :class:`VisionResult` is the *exact* JPEG we
sent to the model — that lets the operator console and the dispatcher
dashboard render the inspected frame next to the verdict, so it's clear
which moment in time the verdict is about.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("tello.vision")

# Plain module-level constant. Vision quality vs cost vs latency was the
# trade-off we tuned for; if you want to A/B another model, change it
# here and re-run smoke_vision.py.
VISION_MODEL = "gpt-4o-mini"
JPEG_QUALITY = 75


class FireDetection(BaseModel):
    """Structured fire/smoke verdict returned by the model."""

    fire_visible: bool = Field(description="True if active flames are visible.")
    smoke_visible: bool = Field(description="True if smoke is visible.")
    confidence: float = Field(
        description="Confidence in the verdict in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    severity: str = Field(
        description="One of: 'none', 'low', 'medium', 'high'.",
    )
    description: str = Field(
        description="One-sentence description of what's in the frame, "
        "including location cues if any.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="1-4 short bullet phrases backing the verdict.",
    )


@dataclass
class VisionResult:
    """Analyzer output, ready to publish on the event bus."""

    fire_visible: bool
    smoke_visible: bool
    confidence: float
    severity: str
    description: str
    reasons: list[str]
    model: str
    thumbnail_b64: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You are a fire/smoke detector for an autonomous response drone.
Look at the camera frame and decide whether there is fire or smoke in this scene.

Rules:
- Be conservative. Indoor false-positive triggers to ignore:
  * red/orange decor, lamps, sunset glare, computer screens, LED strips
  * steam from kettles or bathrooms (looks like smoke but isn't)
  * pictures or videos of fire on screens
- fire_visible = true only when you see actual active flames in the room.
- smoke_visible = true only for gray/black haze that looks like combustion smoke.
- Severity tiers:
  * 'none'   nothing of concern in the frame.
  * 'low'    contained source (candle, small ember).
  * 'medium' small fire or noticeable smoke.
  * 'high'   visible large flames or heavy smoke.
- confidence in [0.0, 1.0] where 1.0 means certain.
- description: a single sentence describing the scene plus any location cues.
- reasons: 1-4 short bullet phrases stating the visual cues that justify the
  verdict (e.g. "orange flicker behind couch", "haze hugging ceiling").
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def encode_jpeg_b64(frame_bgr) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string (no ``data:`` prefix)."""
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    ok, buf = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def analyze_frame(
    frame_bgr,
    *,
    model: str = VISION_MODEL,
    prompt: str | None = None,
) -> VisionResult:
    """Analyze a BGR frame and return a structured fire/smoke verdict.

    Raises:
        ValueError: if ``frame_bgr`` is None.
        RuntimeError: on JPEG encode failure or model refusal.
        Other OpenAI exceptions: bubble up unchanged.
    """
    t0 = time.monotonic()
    thumb_b64 = encode_jpeg_b64(frame_bgr)
    data_url = f"data:image/jpeg;base64,{thumb_b64}"

    user_text = prompt or "Analyze this camera frame for fire or smoke."

    client = _get_client()
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        response_format=FireDetection,
        max_tokens=400,
    )

    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"vision model refused: {msg.refusal}")
    parsed: FireDetection = msg.parsed  # type: ignore[assignment]
    if parsed is None:
        raise RuntimeError("vision model returned no parsed payload")

    return VisionResult(
        fire_visible=bool(parsed.fire_visible),
        smoke_visible=bool(parsed.smoke_visible),
        confidence=float(parsed.confidence),
        severity=str(parsed.severity),
        description=str(parsed.description),
        reasons=list(parsed.reasons),
        model=model,
        thumbnail_b64=thumb_b64,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
