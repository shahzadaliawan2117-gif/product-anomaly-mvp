"""
Production-oriented OCR ensemble for product packaging.

Goals
-----
1. Read text from small, rotated and perspective-affected product images.
2. Never trust a single bad OCR reading for dosage/strength.
3. Keep the public extract_ocr(image) contract compatible with the existing app.
4. Use RapidOCR (PP-OCR models) when installed, with Tesseract as an independent
   fallback/second opinion.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

try:
    from rapidocr import RapidOCR  # rapidocr >= 2
except Exception:  # optional dependency / safe fallback
    RapidOCR = None


DOSAGE_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|µg|ug|ml|cl|l|%)\b",
    re.I,
)
FUZZY_DOSAGE_RE = re.compile(
    # OCR often concatenates a strength to the product name, e.g.
    # "DOLIPRANE500mg" or "DOLIPRANES0O mg".  Do not require a word
    # boundary before the numeric-looking token.
    r"(?<![0-9oOIlLsStT])([0-9oOIlLsStT]{2,7}(?:[.,][0-9oOIlLsStT]+)?)\s*"
    r"(mg|g|mcg|µg|ug|ml|cl|l|%)\b",
    re.I,
)
NUMBER_RE = re.compile(r"\b(\d{2,5})\b")

QUANTITY_RE = re.compile(
    r"(?:\bx\s*(\d{1,4})\b|\b(\d{1,4})\s*"
    r"(?:tablets?|capsules?|comprim[eé]s?|sachets?|units?|pcs?|pieces?|"
    r"g[eé]lules?|ampoules?|doses?)\b)",
    re.I,
)

_DIGIT_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "S": "5",
        "s": "5",
        "T": "1",
        "t": "1",
    }
)

_RAPID_ENGINE = None


def _clean_unit(unit: str) -> str:
    unit = unit.lower()
    return "ug" if unit == "µg" else unit


def _normalise_number(value: str) -> str:
    return value.translate(_DIGIT_MAP).replace(",", ".")


def _canonical_candidate(number: str, unit: str) -> Optional[str]:
    number = _normalise_number(number)
    if not re.fullmatch(r"\d+(?:\.\d+)?", number):
        return None
    try:
        numeric = float(number)
    except ValueError:
        return None
    if numeric <= 0:
        return None
    # absurdly large values are almost always OCR garbage for retail packaging
    if numeric > 100000:
        return None
    return f"{number} {_clean_unit(unit)}"


def _extract_dosages(text: str) -> List[str]:
    if not text:
        return []

    out: List[str] = []
    seen = set()

    for rx in (DOSAGE_RE, FUZZY_DOSAGE_RE):
        for m in rx.finditer(text):
            candidate = _canonical_candidate(m.group(1), m.group(2))
            if candidate and candidate.lower() not in seen:
                seen.add(candidate.lower())
                out.append(candidate)

    return out


# Backward-compatible public helper used by the existing regression suite.
def _extract_dosage_candidates(text: str) -> List[str]:
    return _extract_dosages(text)


# Backward-compatible focused parser used by the pre-ensemble regression tests.
def _extract_focused_dosage_candidates(text: str) -> List[str]:
    return _extract_dosages(text)


def _pick_primary_dosage(votes: Counter, first_seen: List[str]):
    if not votes:
        return None

    unit_priority = {
        "mg": 6, "mcg": 6, "ug": 6, "%": 5,
        "g": 4, "ml": 3, "cl": 2, "l": 2,
    }
    order = {value: index for index, value in enumerate(first_seen)}
    return max(
        votes,
        key=lambda value: (
            votes[value],
            unit_priority.get(value.split()[-1].lower(), 0),
            -order.get(value, 9999),
        ),
    )


def _extract_numbers(text: str) -> List[str]:
    values = []
    seen = set()
    for m in NUMBER_RE.finditer(text or ""):
        value = m.group(1)
        if int(value) <= 0:
            continue
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _rotate_right_angle(image: np.ndarray, angle: int) -> np.ndarray:
    angle %= 360
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Only 0/90/180/270 are supported")


def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    short = min(h, w)
    long = max(h, w)

    # Small product photos benefit from aggressive upscale. This does not
    # recreate missing information, but it materially improves OCR segmentation.
    if short < 900:
        scale = min(4.0, 1200.0 / max(1, short))
    elif long > 3200:
        scale = 3200.0 / long
    else:
        scale = 1.0

    if abs(scale - 1.0) < 0.03:
        return image.copy()

    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def _unsharp(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)


def _preprocess_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    clahe = cv2.createCLAHE(2.5, (8, 8)).apply(gray)
    sharp = _unsharp(clahe)

    _, otsu = cv2.threshold(
        sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    # Individual colour channels can reveal packaging text hidden by coloured
    # backgrounds that grayscale suppresses.
    b, g, r = cv2.split(image)

    return {
        "gray": gray,
        "clahe": clahe,
        "sharp": sharp,
        "otsu": otsu,
        "adaptive": adaptive,
        "red": r,
        "green": g,
        "blue": b,
    }


def _tesseract_text(image: np.ndarray, psm: int) -> Tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        config=f"--oem 3 --psm {psm}",
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

    return " ".join(words).strip(), (float(np.mean(confs)) if confs else 0.0)


def _focused_numeric_text(image: np.ndarray, psm: int) -> str:
    return pytesseract.image_to_string(
        image,
        config=(
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=0123456789mgMGmcgMCGuUµclCL%.,"
        ),
    )


def _rapid_engine():
    global _RAPID_ENGINE

    if RapidOCR is None:
        return None

    if _RAPID_ENGINE is None:
        try:
            _RAPID_ENGINE = RapidOCR()
        except Exception:
            _RAPID_ENGINE = False

    return None if _RAPID_ENGINE is False else _RAPID_ENGINE


def _parse_rapid_result(result: Any) -> List[Tuple[str, float]]:
    """Support both modern RapidOCR result objects and older list/tuple output."""

    if result is None:
        return []

    # Modern RapidOCR result object.
    texts = getattr(result, "txts", None)
    if texts is None:
        texts = getattr(result, "texts", None)

    scores = getattr(result, "scores", None)

    if texts is not None:
        texts = list(texts)
        scores = list(scores) if scores is not None else [0.8] * len(texts)
        out = []
        for text, score in zip(texts, scores):
            text = str(text or "").strip()
            if text:
                try:
                    score_f = float(score)
                except Exception:
                    score_f = 0.8
                out.append((text, 100.0 * score_f if score_f <= 1.0 else score_f))
        return out

    rows = result

    # Older rapidocr_onnxruntime style: (rows, elapsed)
    if isinstance(rows, tuple) and len(rows) >= 1:
        if isinstance(rows[0], (list, tuple)):
            rows = rows[0]

    if not isinstance(rows, (list, tuple)):
        return []

    out: List[Tuple[str, float]] = []

    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue

        # Common shape: [box, text, score]
        if len(row) >= 3:
            text = str(row[1] or "").strip()
            try:
                score = float(row[2])
            except Exception:
                score = 0.8

            if text:
                out.append((text, 100.0 * score if score <= 1.0 else score))

    return out


def _rapid_observations(image: np.ndarray, angle: int) -> List[Dict[str, Any]]:
    engine = _rapid_engine()
    if engine is None:
        return []

    try:
        result = engine(image)
    except Exception:
        return []

    out = []
    for text, confidence in _parse_rapid_result(result):
        out.append(
            {
                "engine": "rapidocr",
                "angle": angle,
                "variant": "color",
                "text": text,
                "confidence": float(confidence),
            }
        )
    return out


def _collect_observations(image: np.ndarray) -> List[Dict[str, Any]]:
    base = _resize_for_ocr(image)
    observations: List[Dict[str, Any]] = []

    rapid_available = _rapid_engine() is not None

    for angle in (0, 90, 180, 270):
        rotated = _rotate_right_angle(base, angle)

        # PP-OCR: text detection + line orientation + recognition.
        if rapid_available:
            observations.extend(_rapid_observations(rotated, angle))

        variants = _preprocess_variants(rotated)

        # Independent Tesseract evidence. Keep the pass count controlled.
        tess_variants = ("clahe", "sharp", "adaptive")
        tess_psms = (6, 11)

        for variant_name in tess_variants:
            variant = variants[variant_name]
            for psm in tess_psms:
                try:
                    text, confidence = _tesseract_text(variant, psm)
                except Exception:
                    continue

                if text:
                    observations.append(
                        {
                            "engine": "tesseract",
                            "angle": angle,
                            "variant": f"{variant_name}-psm{psm}",
                            "text": text,
                            "confidence": confidence,
                        }
                    )

        # Dosage-only pass on the two strongest representations.
        for variant_name in ("sharp", "adaptive"):
            for psm in (6, 11, 12):
                try:
                    text = _focused_numeric_text(variants[variant_name], psm)
                except Exception:
                    continue

                text = " ".join(text.split())
                if text:
                    observations.append(
                        {
                            "engine": "tesseract_numeric",
                            "angle": angle,
                            "variant": f"{variant_name}-psm{psm}",
                            "text": text,
                            "confidence": 62.0,
                        }
                    )

    return observations


def _unique_support_key(obs: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(obs.get("engine", "")),
        int(obs.get("angle", 0)),
        str(obs.get("variant", "")),
    )


def _rank_dosage_candidates(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, int], Dict[str, float], bool]:
    support_sets: Dict[str, set] = defaultdict(set)
    score: Dict[str, float] = defaultdict(float)

    for obs in observations:
        text = str(obs.get("text", ""))
        conf = float(obs.get("confidence", 0.0))
        key = _unique_support_key(obs)

        for candidate in set(_extract_dosages(text)):
            support_sets[candidate].add(key)

            # Confidence is deliberately capped so many independent passes are
            # more valuable than one overconfident OCR hallucination.
            quality = max(0.20, min(1.0, conf / 100.0))
            engine_bonus = 1.20 if obs.get("engine") == "rapidocr" else 1.0
            score[candidate] += quality * engine_bonus

    if not score:
        return None, {}, {}, False

    def rank_key(candidate: str):
        support = len(support_sets[candidate])
        return (support, score[candidate])

    best = max(score, key=rank_key)
    best_support = len(support_sets[best])
    best_score = score[best]

    # Require consensus. This is the key safeguard against "9 g", "00 mg",
    # and similar one-pass false reads.
    reliable = (
        best_support >= 3
        or (best_support >= 2 and best_score >= 1.20)
    )

    votes = {k: len(v) for k, v in support_sets.items()}
    scores = {k: round(v, 3) for k, v in score.items()}

    return (best if reliable else None), votes, scores, reliable


def _number_votes(observations: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    support: Dict[str, set] = defaultdict(set)

    for obs in observations:
        if obs.get("engine") != "tesseract_numeric":
            continue

        key = _unique_support_key(obs)

        for number in set(_extract_numbers(str(obs.get("text", "")))):
            support[number].add(key)

    return {number: len(keys) for number, keys in support.items()}


def _merge_text(observations: Sequence[Dict[str, Any]]) -> str:
    if not observations:
        return ""

    # Put high-confidence recognitions first but keep distinct alternate reads.
    ranked = sorted(
        observations,
        key=lambda x: (
            float(x.get("confidence", 0.0)),
            len(str(x.get("text", ""))),
        ),
        reverse=True,
    )

    chunks: List[str] = []
    seen = set()

    for obs in ranked:
        text = " ".join(str(obs.get("text", "")).split()).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        chunks.append(text)

        # Keep response diagnostics useful without creating megabytes of text.
        if len(" | ".join(chunks)) > 8000:
            break

    return " | ".join(chunks)


def _ocr_confidence(observations: Sequence[Dict[str, Any]]) -> float:
    useful = [
        float(obs.get("confidence", 0.0))
        for obs in observations
        if obs.get("engine") != "tesseract_numeric"
        and str(obs.get("text", "")).strip()
    ]

    if not useful:
        return 0.0

    useful.sort(reverse=True)
    return round(float(np.mean(useful[: min(8, len(useful))])), 2)


def extract_ocr(image: np.ndarray) -> Dict[str, Any]:
    """Run multi-engine, multi-orientation OCR.

    The function remains compatible with the existing pipeline, while exposing
    additional reliability diagnostics that product_matching.py can use.
    """

    try:
        observations = _collect_observations(image)
    except Exception as exc:
        return {
            "text": "",
            "confidence": 0.0,
            "dosage": None,
            "dosage_candidates": [],
            "dosage_candidate_votes": {},
            "dosage_candidate_scores": {},
            "dosage_number_votes": {},
            "dosage_reliable": False,
            "ocr_reliable": False,
            "quantity": None,
            "engines": [],
            "error": str(exc),
        }

    text = _merge_text(observations)
    confidence = _ocr_confidence(observations)

    dosage, dosage_votes, dosage_scores, dosage_reliable = (
        _rank_dosage_candidates(observations)
    )
    number_votes = _number_votes(observations)

    quantity = None
    q = QUANTITY_RE.search(text)
    if q:
        quantity = q.group(1) or q.group(2)

    engines = sorted(
        {
            str(obs.get("engine"))
            for obs in observations
            if str(obs.get("text", "")).strip()
        }
    )

    token_count = len(re.findall(r"[A-Za-z0-9%]+", text))
    ocr_reliable = (
        confidence >= 45.0
        and token_count >= 3
        and len(observations) >= 2
    )

    return {
        "text": text,
        "confidence": confidence,
        "dosage": dosage,
        "dosage_candidates": sorted(
            dosage_votes,
            key=lambda c: (
                -dosage_votes[c],
                -dosage_scores.get(c, 0.0),
                c,
            ),
        ),
        "dosage_candidate_votes": dosage_votes,
        "dosage_candidate_scores": dosage_scores,
        "dosage_number_votes": number_votes,
        "dosage_reliable": dosage_reliable,
        "ocr_reliable": ocr_reliable,
        "quantity": quantity,
        "engines": engines,
        "observation_count": len(observations),
        "error": None,
    }
