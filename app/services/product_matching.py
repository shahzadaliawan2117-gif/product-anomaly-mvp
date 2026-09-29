import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Any, Optional
from app.models.schemas import ProductMetadata


def norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9%]+", " ", s).strip()


def tokens(s: Optional[str]):
    return [x for x in norm(s).split() if len(x) > 1]


def _token_close(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.84


def fuzzy(expected: Optional[str], text: Optional[str]):
    """Token-coverage matcher that is conservative on unrelated products.

    The previous whole-string SequenceMatcher could give surprisingly high
    scores to short unrelated OCR strings (for example PANTENE vs SANOFI).
    """
    e = norm(expected)
    t = norm(text)
    if not e:
        return None
    if not t:
        return 0.0
    if e in t:
        return 1.0

    et = tokens(e)
    tt = tokens(t)
    if not et:
        return 0.0

    hits = 0.0
    for token in et:
        if token in tt:
            hits += 1.0
        elif any(_token_close(token, candidate) for candidate in tt):
            hits += 0.75
    return min(1.0, hits / len(et))


def exact_visible(expected: Optional[str], text: str) -> bool:
    e = norm(expected)
    t = norm(text)
    if not e or not t:
        return False
    if e in t:
        return True
    et = tokens(e)
    tt = tokens(t)
    return bool(et) and all(any(_token_close(x, y) for y in tt) for x in et)


def dosage_value(s):
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|ml|%)", norm(s))
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    # Canonicalize mass units so OCR readings such as 1 g and metadata
    # such as 1000 mg are treated as the same strength.
    if unit == "g":
        return (round(value * 1000, 4), "mg")
    if unit in {"mcg", "ug"}:
        return (round(value / 1000, 4), "mg")
    return (round(value, 4), unit)


def infer_category(text: str) -> Optional[str]:
    t = norm(text)
    tt = set(tokens(t))

    # High-specificity personal-care cues. One strong cue is enough because
    # OCR may only recover a brand or the word "shampoo" from stylized labels.
    shampoo_strong = {
        "shampoo",
        "conditioner",
        "pantene",
        "dandruff",
    }
    shampoo_support = {"hair", "silky", "scalp", "aloe", "nutrients"}
    if tt & shampoo_strong or len(tt & shampoo_support) >= 2:
        return "Shampoo"

    # Medicine/pharmacy cues. Avoid treating a bare number/unit as enough.
    medicine_strong = {
        "paracetamol",
        "ibuprofen",
        "doliprane",
        "maalox",
        "antacid",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "comprime",
        "comprimes",
        "pharma",
        "medicine",
    }
    if tt & medicine_strong:
        return "Medicament"

    return None


def match_product(meta: ProductMetadata, ocr: Dict[str, Any]) -> Dict[str, Any]:
    text = ocr.get("text", "")
    fields = {}
    for key in ["name", "brand", "laboratory", "form", "packaging", "variant"]:
        value = getattr(meta, key, None)
        if value:
            fields[key] = fuzzy(value, text)

    expected_d = dosage_value(meta.dosage or meta.strength or "")
    detected_d = dosage_value(ocr.get("dosage") or text)
    dosage_match = None
    if expected_d and detected_d:
        dosage_match = 1.0 if expected_d == detected_d else 0.0

    detected_category = infer_category(text)
    category_match = None
    if meta.category and detected_category:
        category_match = fuzzy(meta.category, detected_category)

    evidence = []
    for key, weight in [
        ("name", 0.48),
        ("brand", 0.22),
        ("laboratory", 0.08),
        ("form", 0.08),
        ("packaging", 0.06),
        ("variant", 0.08),
    ]:
        if key in fields:
            evidence.append((fields[key], weight))
    if dosage_match is not None:
        evidence.append((dosage_match, 0.28))
    if category_match is not None:
        evidence.append((category_match, 0.12))

    if evidence:
        score = 100 * sum(v * w for v, w in evidence) / sum(w for _, w in evidence)
    else:
        score = 50.0

    readable = ocr.get("confidence", 0) >= 35 and len(tokens(text)) >= 2
    # Unreadable text means uncertainty, not a product mismatch.
    if not readable:
        score = max(score, 50.0)

    name_score = fields.get("name")
    brand_score = fields.get("brand")
    status = None
    reason = None

    # A variant/brand mismatch is only meaningful when the underlying product
    # still appears to be the same product family.
    same_product = name_score is None or name_score >= 0.55

    # Precedence: category -> clearly different product -> brand -> variant.
    if category_match is not None and category_match < 0.55:
        status = "WRONG_CATEGORY"
        reason = f"Detected category '{detected_category}' conflicts with expected category '{meta.category}'."
    elif name_score is not None and name_score < 0.42 and readable:
        status = "WRONG_PRODUCT"
        reason = "Visible product text strongly conflicts with the expected product."
    elif brand_score is not None and brand_score < 0.35 and readable and same_product:
        status = "WRONG_BRAND"
        reason = "Visible brand text conflicts with the expected brand."
    elif (
        dosage_match == 0.0
        or (fields.get("variant") is not None and fields["variant"] < 0.35)
    ) and same_product:
        status = "WRONG_VARIANT"
        reason = "Visible variant/dosage conflicts with expected metadata."

    detected_brand = meta.brand if exact_visible(meta.brand, text) else None
    detected_name = meta.name if name_score is not None and name_score >= 0.65 else None
    detected_form = meta.form if exact_visible(meta.form, text) else None

    return {
        "product_match": round(score, 1),
        "mismatch_status": status,
        "reason": reason,
        "field_scores": {k: round(v * 100, 1) for k, v in fields.items()},
        "dosage_match": None if dosage_match is None else bool(dosage_match),
        "detected_category": detected_category,
        "evidence_reliable": readable,
        "detected": {
            "name": detected_name,
            "brand": detected_brand,
            "category": detected_category,
            "dosage": ocr.get("dosage"),
            "form": detected_form,
            "quantity": ocr.get("quantity"),
        },
    }
