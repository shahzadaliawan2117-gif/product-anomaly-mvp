import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import ProductMetadata


DOSAGE_TOKEN_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|µg|ml|cl|l|%)",
    re.I,
)


def norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
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


def dosage_value(s: Optional[str]) -> Optional[Tuple[float, str]]:
    if not s:
        return None

    match = DOSAGE_TOKEN_RE.search(str(s))
    if not match:
        return None

    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower().replace("µg", "ug")

    if unit == "g":
        return (round(value * 1000, 4), "mg")
    if unit in {"mcg", "ug"}:
        return (round(value / 1000, 4), "mg")
    if unit == "cl":
        return (round(value * 10, 4), "ml")
    if unit == "l":
        return (round(value * 1000, 4), "ml")

    return (round(value, 4), unit)


def dosage_family(value: Optional[Tuple[float, str]]) -> Optional[str]:
    if not value:
        return None

    unit = value[1]
    if unit == "mg":
        return "mass"
    if unit == "ml":
        return "volume"
    if unit == "%":
        return "percent"
    return unit


def _expected_dosage(meta: ProductMetadata) -> Optional[Tuple[float, str]]:
    # Strength may exist only in the product name, e.g.
    # "Doliprane 1000 mg Comprime".
    for value in [
        meta.dosage,
        meta.strength,
        meta.variant,
        meta.name,
        meta.packaging,
    ]:
        parsed = dosage_value(value)
        if parsed:
            return parsed

    return None


def _detected_dosage_evidence(
    ocr: Dict[str, Any],
    expected: Optional[Tuple[float, str]],
) -> Tuple[
    Optional[Tuple[float, str]],
    Optional[str],
    List[Tuple[float, str]],
    Dict[str, int],
]:
    raw_candidates: List[str] = []

    if ocr.get("dosage"):
        raw_candidates.append(str(ocr["dosage"]))

    for value in ocr.get("dosage_candidates") or []:
        if value and str(value) not in raw_candidates:
            raw_candidates.append(str(value))

    if not raw_candidates and ocr.get("text"):
        for match in DOSAGE_TOKEN_RE.finditer(str(ocr["text"])):
            raw = f"{match.group(1)} {match.group(2)}"
            if raw not in raw_candidates:
                raw_candidates.append(raw)

    votes = {
        str(key): int(value)
        for key, value in (ocr.get("dosage_candidate_votes") or {}).items()
        if isinstance(value, (int, float))
    }

    parsed_candidates: List[Tuple[float, str]] = []
    parsed_to_raw: Dict[Tuple[float, str], str] = {}

    for raw in raw_candidates:
        parsed = dosage_value(raw)
        if parsed and parsed not in parsed_candidates:
            parsed_candidates.append(parsed)
            parsed_to_raw[parsed] = raw

    if not parsed_candidates:
        return None, None, [], votes

    comparable = parsed_candidates

    if expected:
        expected_family = dosage_family(expected)
        same_family = [
            value
            for value in parsed_candidates
            if dosage_family(value) == expected_family
        ]

        # A package volume such as 200 ml must not be compared to a tablet
        # strength such as 1000 mg.
        if same_family:
            comparable = same_family
        else:
            return None, None, parsed_candidates, votes

    primary_raw = str(ocr.get("dosage") or "")
    primary = dosage_value(primary_raw)

    if primary in comparable:
        return primary, primary_raw, parsed_candidates, votes

    chosen = comparable[0]
    return chosen, parsed_to_raw.get(chosen), parsed_candidates, votes


def infer_category(text: str) -> Optional[str]:
    t = norm(text)
    tt = set(tokens(t))

    shampoo_strong = {"shampoo", "conditioner", "pantene", "dandruff"}
    shampoo_support = {"hair", "silky", "scalp", "aloe", "nutrients"}

    if tt & shampoo_strong or len(tt & shampoo_support) >= 2:
        return "Shampoo"

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

    expected_d = _expected_dosage(meta)

    (
        detected_d,
        detected_d_raw,
        detected_d_candidates,
        detected_d_votes,
    ) = _detected_dosage_evidence(ocr, expected_d)

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

    if not readable:
        score = max(score, 50.0)

    name_score = fields.get("name")
    brand_score = fields.get("brand")
    status = None
    reason = None

    same_product = name_score is None or name_score >= 0.55

    if category_match is not None and category_match < 0.55:
        status = "WRONG_CATEGORY"
        reason = (
            f"Detected category '{detected_category}' conflicts with "
            f"expected category '{meta.category}'."
        )
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

        expected_display = (
            meta.dosage
            or meta.strength
            or (f"{expected_d[0]:g} {expected_d[1]}" if expected_d else None)
        )

        detected_display = (
            detected_d_raw
            or (f"{detected_d[0]:g} {detected_d[1]}" if detected_d else "unknown")
        )

        reason = (
            f"Visible variant/dosage '{detected_display}' conflicts with "
            f"expected dosage '{expected_display}'."
        )

    detected_brand = meta.brand if exact_visible(meta.brand, text) else None
    detected_name = meta.name if name_score is not None and name_score >= 0.65 else None
    detected_form = meta.form if exact_visible(meta.form, text) else None

    candidate_display = [
        f"{value:g} {unit}"
        for value, unit in detected_d_candidates
    ]

    return {
        "product_match": round(score, 1),
        "mismatch_status": status,
        "reason": reason,
        "field_scores": {k: round(v * 100, 1) for k, v in fields.items()},
        "expected_dosage": (
            None if expected_d is None else f"{expected_d[0]:g} {expected_d[1]}"
        ),
        "dosage_match": None if dosage_match is None else bool(dosage_match),
        "detected_dosage_candidates": candidate_display,
        "detected_dosage_votes": detected_d_votes,
        "detected_category": detected_category,
        "evidence_reliable": readable,
        "detected": {
            "name": detected_name,
            "brand": detected_brand,
            "category": detected_category,
            "dosage": detected_d_raw or ocr.get("dosage"),
            "form": detected_form,
            "quantity": ocr.get("quantity"),
        },
    }
