from typing import Any, Dict, List, Optional, Tuple

from app.services.product_matching import dosage_value


def _uncertain(vision_fields: Dict[str, Any], key: str) -> bool:
    return key in set(vision_fields.get("uncertain_fields") or [])


def _visible_strings(vision_fields: Dict[str, Any]) -> List[str]:
    return [
        str(x).strip()
        for x in (vision_fields.get("visible_text") or [])
        if str(x).strip()
    ]


def _vision_visible_dosages(
    vision_fields: Dict[str, Any],
) -> List[Tuple[Tuple[float, str], str]]:
    values: List[Tuple[Tuple[float, str], str]] = []
    seen = set()

    for text in _visible_strings(vision_fields):
        parsed = dosage_value(text)
        if parsed and parsed not in seen:
            seen.add(parsed)
            values.append((parsed, text))

    return values


def _trusted_vision_dosage(
    vision_fields: Dict[str, Any],
) -> Tuple[Optional[Tuple[float, str]], Optional[str]]:
    """Trust dosage only when the VLM's structured field is visually grounded.

    Important:
    - If the structured dosage is marked uncertain BUT the exact same dosage is
      independently present in visible_text, accept it.
    - If the structured field is missing but visible_text contains exactly one
      dosage value, use that visible dosage.
    - If visible_text contains conflicting dosage values, return unknown.
    """
    visible = _vision_visible_dosages(vision_fields)
    visible_values = [parsed for parsed, _ in visible]

    raw = vision_fields.get("dosage_strength")
    parsed_raw = dosage_value(str(raw)) if raw else None

    if parsed_raw is not None and parsed_raw in visible_values:
        # visible_text corroboration is stronger than the model's own
        # uncertain_fields bookkeeping.
        return parsed_raw, str(raw)

    if raw and not _uncertain(vision_fields, "dosage_strength"):
        # A confident structured field still needs visual grounding. If the
        # model did not quote the dosage in visible_text, do not trust it.
        return None, None

    unique_visible = []
    for parsed, text in visible:
        if parsed not in [x[0] for x in unique_visible]:
            unique_visible.append((parsed, text))

    if len(unique_visible) == 1:
        parsed, text = unique_visible[0]
        return parsed, text

    return None, None


def _ocr_dosage(ocr: Dict[str, Any]):
    raw = ocr.get("dosage")
    return dosage_value(raw) if raw else None


def _identity_fields(vision_fields: Dict[str, Any]) -> List[str]:
    values = []

    for key in (
        "product_name",
        "brand",
        "manufacturer",
        "form",
        "category",
        "variant",
    ):
        value = vision_fields.get(key)
        if value and not _uncertain(vision_fields, key):
            values.append(str(value))

    return values


def _trusted_visible_field(
    vision_fields: Dict[str, Any],
    key: str,
) -> Optional[str]:
    """Return a field only when the VLM is confident and visually grounds it.

    This is deliberately strict for identity fields that can create hard
    mismatch statuses such as WRONG_BRAND.
    """
    value = vision_fields.get(key)
    if not value or _uncertain(vision_fields, key):
        return None

    value_text = str(value).strip()
    if not value_text:
        return None

    value_lower = value_text.lower()

    # For brand/manufacturer, require it to appear in the quoted visible text.
    if key in {"brand", "manufacturer"}:
        visible = " | ".join(_visible_strings(vision_fields)).lower()
        if value_lower not in visible:
            return None

    return value_text


def resolve_evidence(
    ocr: Dict[str, Any],
    vision: Dict[str, Any],
) -> Dict[str, Any]:
    """Return OCR-compatible hybrid evidence for match_product()."""
    resolved = dict(ocr)

    if not vision.get("available"):
        resolved["hybrid_source"] = "ocr_only"
        resolved["vision_conflict"] = None
        resolved["observed_brand"] = None
        return resolved

    vf = vision.get("fields") or {}

    identity = _identity_fields(vf)
    visible_text = _visible_strings(vf)

    parts = []
    if ocr.get("text"):
        parts.append(str(ocr["text"]))
    parts.extend(visible_text)
    parts.extend(identity)

    resolved["text"] = " | ".join(dict.fromkeys(parts))

    vision_dosage, vision_raw = _trusted_vision_dosage(vf)
    ocr_dosage = _ocr_dosage(ocr)
    ocr_reliable = bool(ocr.get("dosage_reliable", False))

    conflict = None

    if vision_dosage is not None:
        if ocr_dosage is not None and ocr_dosage != vision_dosage:
            conflict = {
                "ocr": ocr.get("dosage"),
                "vision": vision_raw,
            }

        resolved["dosage"] = vision_raw
        resolved["dosage_candidates"] = [vision_raw]
        resolved["dosage_candidate_votes"] = {vision_raw: 6}
        resolved["dosage_candidate_scores"] = {vision_raw: 6.0}
        resolved["dosage_reliable"] = True

    elif ocr_dosage is not None and ocr_reliable:
        resolved["dosage"] = ocr.get("dosage")

    else:
        resolved["dosage"] = None
        resolved["dosage_candidates"] = []
        resolved["dosage_candidate_votes"] = {}
        resolved["dosage_candidate_scores"] = {}
        resolved["dosage_reliable"] = False

    if identity:
        resolved["ocr_reliable"] = True
        resolved["confidence"] = max(
            float(ocr.get("confidence", 0) or 0),
            82.0,
        )

    if not resolved.get("quantity") and vf.get("quantity") and not _uncertain(vf, "quantity"):
        resolved["quantity"] = str(vf["quantity"])

    # Explicit brand evidence is separate from free OCR text. The matcher may
    # only emit WRONG_BRAND from this trusted observation, not from "brand text
    # was absent from OCR".
    resolved["observed_brand"] = _trusted_visible_field(vf, "brand")

    resolved["hybrid_source"] = "ocr_plus_local_vision"
    resolved["vision_conflict"] = conflict
    resolved["vision_fields"] = vf

    return resolved
