import base64
import json
import os
import re
from typing import Any, Dict

import cv2
import requests


DEFAULT_MODEL = "hf.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
VISION_MODEL = os.getenv("VISION_MODEL", DEFAULT_MODEL)
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT_SECONDS", "180"))

PROMPT = """You are a product-packaging visual inspection engine.

Inspect ONLY the supplied image. Do not use any expected product metadata.
Read only information that is genuinely visible. If a field is not readable,
return null instead of guessing.

Return JSON only, with exactly these keys:
{
  "product_name": string|null,
  "brand": string|null,
  "manufacturer": string|null,
  "dosage_strength": string|null,
  "form": string|null,
  "quantity": string|null,
  "category": string|null,
  "variant": string|null,
  "visible_text": [string],
  "uncertain_fields": [string]
}

Rules:
- Dosage/strength and package quantity are different.
- Preserve visible units: 200 mg, 500 mg, 1 g, 100 ml, etc.
- Do not turn package counts into dosage.
- visible_text must contain short exact strings you can actually see.
- If dosage is readable, include the exact dosage-bearing text in visible_text.
- If a field is doubtful, add its key to uncertain_fields and set it to null.
"""


def vision_enabled() -> bool:
    return os.getenv("LOCAL_VISION_ENABLED", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Vision model did not return valid JSON")

    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Vision model JSON was not an object")
    return value


def _normalise_result(value: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "product_name",
        "brand",
        "manufacturer",
        "dosage_strength",
        "form",
        "quantity",
        "category",
        "variant",
    ]

    result = {}
    for key in keys:
        item = value.get(key)
        if item is None:
            result[key] = None
        else:
            text = str(item).strip()
            result[key] = text or None

    visible = value.get("visible_text") or []
    if not isinstance(visible, list):
        visible = [visible]
    result["visible_text"] = [
        str(x).strip() for x in visible if str(x).strip()
    ]

    uncertain = value.get("uncertain_fields") or []
    if not isinstance(uncertain, list):
        uncertain = [uncertain]
    result["uncertain_fields"] = [
        str(x).strip() for x in uncertain if str(x).strip()
    ]

    return result


def inspect_image(image) -> Dict[str, Any]:
    """Inspect an already-decoded OpenCV BGR image with the local Ollama VLM.

    The expected Product JSON is intentionally NOT sent to the model, which
    prevents the verifier from being biased toward the expected answer.
    """
    if not vision_enabled():
        return {
            "available": False,
            "enabled": False,
            "fields": {},
            "error": None,
        }

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 94],
    )
    if not ok:
        return {
            "available": False,
            "enabled": True,
            "fields": {},
            "error": "Could not encode image for local vision model",
        }

    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            }
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=VISION_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        raw = body.get("message", {}).get("content", "")
        fields = _normalise_result(_extract_json(raw))

        return {
            "available": True,
            "enabled": True,
            "model": VISION_MODEL,
            "fields": fields,
            "error": None,
        }

    except Exception as exc:
        # Vision is an optional verifier. A local Ollama failure must not take
        # the whole API down; the existing OCR/CV pipeline remains available.
        return {
            "available": False,
            "enabled": True,
            "model": VISION_MODEL,
            "fields": {},
            "error": str(exc),
        }
