import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

DOSAGE_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|µg|ug|ml|cl|l|%)\b", re.I)
FUZZY_DOSAGE_RE = re.compile(
    r"\b([0-9oOIlLsS]{1,7}(?:[.,][0-9oOIlLsS]+)?)\s*(mg|g|mcg|µg|ug|ml|cl|l|%)\b",
    re.I,
)
QUANTITY_RE = re.compile(
    r"(?:\bx\s*(\d{1,4})\b|\b(\d{1,4})\s*(?:tablets?|capsules?|comprim[eé]s?|sachets?|units?|pcs?|pieces?)\b)",
    re.I,
)

_OCR_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "i": "1",
    "L": "1", "l": "1",
    "S": "5", "s": "5",
})


def _normalise_numeric_token(value: str) -> str:
    return value.translate(_OCR_DIGIT_MAP).replace(",", ".")


def _clean_unit(unit: str) -> str:
    unit = unit.lower()
    return "ug" if unit == "µg" else unit


def _extract_dosage_candidates(text: str) -> List[str]:
    if not text:
        return []

    found: List[str] = []
    seen = set()

    for match in DOSAGE_RE.finditer(text):
        candidate = f"{match.group(1).replace(',', '.')} {_clean_unit(match.group(2))}"
        if candidate.lower() not in seen:
            found.append(candidate)
            seen.add(candidate.lower())

    for match in FUZZY_DOSAGE_RE.finditer(text):
        number = _normalise_numeric_token(match.group(1))
        if not re.fullmatch(r"\d+(?:\.\d+)?", number):
            continue
        candidate = f"{number} {_clean_unit(match.group(2))}"
        if candidate.lower() not in seen:
            found.append(candidate)
            seen.add(candidate.lower())

    return found


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


def _pick_primary_dosage(votes: Counter, first_seen: List[str]):
    if not votes:
        return None

    unit_priority = {"mg": 4, "g": 4, "mcg": 4, "ug": 4, "%": 3, "ml": 2, "cl": 2, "l": 2}
    order = {value: index for index, value in enumerate(first_seen)}

    return max(
        votes,
        key=lambda value: (
            votes[value],
            unit_priority.get(value.split()[-1].lower(), 0),
            -order.get(value, 9999),
        ),
    )


def extract_ocr(image: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    scale = max(1.0, 1800 / max(gray.shape))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    thresholded = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    passes = [
        (gray, 6),
        (gray, 11),
        (gray, 3),
        (thresholded, 11),
        (thresholded, 3),
    ]

    try:
        results = [_ocr_pass(img, psm) for img, psm in passes]
        valid = [result for result in results if result[0]]

        if not valid:
            return {
                "text": "",
                "confidence": 0.0,
                "dosage": None,
                "dosage_candidates": [],
                "dosage_candidate_votes": {},
                "quantity": None,
                "error": None,
            }

        text = _merge_texts(valid)
        best = max(valid, key=lambda r: (r[2], r[1]))
        mean_conf = best[1]

    except Exception as exc:
        return {
            "text": "",
            "confidence": 0.0,
            "dosage": None,
            "dosage_candidates": [],
            "dosage_candidate_votes": {},
            "quantity": None,
            "error": str(exc),
        }

    dosage_votes: Counter = Counter()
    first_seen: List[str] = []

    for pass_text, _, _ in valid:
        for candidate in dict.fromkeys(_extract_dosage_candidates(pass_text)):
            dosage_votes[candidate] += 1
            if candidate not in first_seen:
                first_seen.append(candidate)

    for candidate in _extract_dosage_candidates(text):
        if candidate not in dosage_votes:
            dosage_votes[candidate] = 1
            first_seen.append(candidate)

    dosage = _pick_primary_dosage(dosage_votes, first_seen)
    dosage_candidates = sorted(
        dosage_votes,
        key=lambda value: (-dosage_votes[value], first_seen.index(value)),
    )

    quantity = None
    q = QUANTITY_RE.search(text)
    if q:
        quantity = q.group(1) or q.group(2)

    return {
        "text": text,
        "confidence": round(mean_conf, 2),
        "dosage": dosage,
        "dosage_candidates": dosage_candidates,
        "dosage_candidate_votes": dict(dosage_votes),
        "quantity": quantity,
        "error": None,
    }
