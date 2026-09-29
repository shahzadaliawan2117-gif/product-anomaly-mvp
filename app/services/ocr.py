import re
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pytesseract

DOSAGE_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|µg|ug|ml|cl|l|%)\b", re.I)
QUANTITY_RE = re.compile(
    r"(?:\bx\s*(\d{1,4})\b|\b(\d{1,4})\s*(?:tablets?|capsules?|comprim[eé]s?|sachets?|units?|pcs?|pieces?)\b)",
    re.I,
)


def _ocr_pass(image: np.ndarray, psm: int) -> Tuple[str, float, int]:
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    words: List[str] = []
    confs: List[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = (text or "").strip()
        try:
            cf = float(conf)
        except (TypeError, ValueError):
            cf = -1
        if text and cf >= 0:
            words.append(text)
            confs.append(cf)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return " ".join(words).strip(), mean_conf, len(words)


def _merge_texts(results: List[Tuple[str, float, int]]) -> str:
    # Keep the strongest pass first, then add useful text found by other layouts.
    ranked = sorted(results, key=lambda r: (r[2], r[1]), reverse=True)
    chunks: List[str] = []
    seen = set()
    for text, _, _ in ranked:
        cleaned = " ".join(text.split()).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            chunks.append(cleaned)
            seen.add(key)
    return " | ".join(chunks)


def extract_ocr(image: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale packaging text so small labels are more OCR-friendly.
    scale = max(1.0, 1800 / max(gray.shape))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    thresholded = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Product packaging varies a lot. Multiple deterministic OCR layouts are
    # substantially more reliable than relying on a single PSM mode.
    passes = [
        (gray, 6),
        (gray, 11),
        (gray, 3),
        (thresholded, 11),
        (thresholded, 3),
    ]

    try:
        results = [_ocr_pass(img, psm) for img, psm in passes]
        valid = [r for r in results if r[0]]
        if not valid:
            return {"text": "", "confidence": 0.0, "dosage": None, "quantity": None, "error": None}

        text = _merge_texts(valid)
        # Use the confidence of the strongest text-bearing pass rather than
        # averaging weak alternative layouts into the score.
        best = max(valid, key=lambda r: (r[2], r[1]))
        mean_conf = best[1]
    except Exception as exc:
        return {"text": "", "confidence": 0.0, "dosage": None, "quantity": None, "error": str(exc)}

    dosage = None
    dosage_matches = DOSAGE_RE.findall(text)
    if dosage_matches:
        # Prefer mg/g/mcg strengths over package-volume ml when both exist.
        preferred = next((m for m in dosage_matches if m[1].lower() in {"mg", "g", "mcg", "µg", "ug", "%"}), dosage_matches[0])
        dosage = f"{preferred[0]} {preferred[1]}"

    quantity = None
    q = QUANTITY_RE.search(text)
    if q:
        quantity = q.group(1) or q.group(2)

    return {
        "text": text,
        "confidence": round(mean_conf, 2),
        "dosage": dosage,
        "quantity": quantity,
        "error": None,
    }
