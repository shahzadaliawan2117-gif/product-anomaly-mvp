import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import ProductMetadata


DOSAGE_TOKEN_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|µg|ml|cl|l|%)",
    re.I,
)

GENERIC_PRODUCT_TOKENS = {
    "mg", "g", "mcg", "ug", "ml", "cl", "l",
    "tablet", "tablets", "comprime", "comprimes",
    "capsule", "capsules", "gelule", "gelules",
    "boite", "box", "pack", "oral", "dose",
}


def norm(value: Optional[str]) -> str:
    if not value:
        return ""
    value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def tokens(value: Optional[str]) -> List[str]:
    return [x for x in norm(value).split() if len(x) > 1]


def _token_close(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) >= 5 and (a in b or b in a):
        return True
    if min(len(a), len(b)) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.84


def fuzzy(expected: Optional[str], text: Optional[str]):
    expected_norm = norm(expected)
    text_norm = norm(text)

    if not expected_norm:
        return None
    if not text_norm:
        return 0.0
    if expected_norm in text_norm:
        return 1.0

    expected_tokens = tokens(expected_norm)
    visible_tokens = tokens(text_norm)

    if not expected_tokens:
        return 0.0

    hits = 0.0
    for token in expected_tokens:
        if token in visible_tokens:
            hits += 1.0
        elif any(_token_close(token, candidate) for candidate in visible_tokens):
            hits += 0.75

    return min(1.0, hits / len(expected_tokens))


def exact_visible(expected: Optional[str], text: str) -> bool:
    expected_norm = norm(expected)
    text_norm = norm(text)

    if not expected_norm or not text_norm:
        return False
    if expected_norm in text_norm:
        return True

    return all(
        any(_token_close(x, y) for y in tokens(text_norm))
        for x in tokens(expected_norm)
    )


def dosage_value(value: Optional[str]) -> Optional[Tuple[float, str]]:
    if not value:
        return None

    m = DOSAGE_TOKEN_RE.search(str(value))
    if not m:
        return None

    number = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower().replace("µg", "ug")

    if number <= 0:
        return None

    if unit == "g":
        return round(number * 1000, 4), "mg"
    if unit in {"mcg", "ug"}:
        return round(number / 1000, 4), "mg"
    if unit == "cl":
        return round(number * 10, 4), "ml"
    if unit == "l":
        return round(number * 1000, 4), "ml"

    return round(number, 4), unit


def dosage_family(value: Optional[Tuple[float, str]]) -> Optional[str]:
    if not value:
        return None
    if value[1] == "mg":
        return "mass"
    if value[1] == "ml":
        return "volume"
    if value[1] == "%":
        return "percent"
    return value[1]


def _expected_dosage(meta: ProductMetadata) -> Optional[Tuple[float, str]]:
    for value in (
        meta.dosage,
        meta.strength,
        meta.variant,
        meta.name,
        meta.packaging,
    ):
        parsed = dosage_value(value)
        if parsed:
            return parsed
    return None


def _core_name_tokens(name: Optional[str]) -> List[str]:
    if not name:
        return []

    cleaned = DOSAGE_TOKEN_RE.sub(" ", str(name))
    return [
        token
        for token in tokens(cleaned)
        if token not in GENERIC_PRODUCT_TOKENS
        and not token.isdigit()
    ]


def _core_product_score(name: Optional[str], text: str) -> Optional[float]:
    expected = _core_name_tokens(name)
    if not expected:
        return None

    visible = tokens(text)
    if not visible:
        return 0.0

    hits = 0.0
    for token in expected:
        if token in visible:
            hits += 1.0
        elif any(_token_close(token, candidate) for candidate in visible):
            hits += 0.75

    return min(1.0, hits / len(expected))


def infer_category(text: str) -> Optional[str]:
    visible = set(tokens(text))

    shampoo_strong = {
        "shampoo", "conditioner", "pantene", "dandruff",
    }
    shampoo_support = {
        "hair", "scalp", "silky", "aloe", "nutrients",
    }

    if visible & shampoo_strong or len(visible & shampoo_support) >= 2:
        return "Shampoo"

    medicine_strong = {
        "paracetamol", "ibuprofen", "doliprane", "maalox",
        "antacid", "tablet", "tablets", "capsule", "capsules",
        "comprime", "comprimes", "gelule", "gelules",
        "pharma", "medicine",
    }

    if visible & medicine_strong:
        return "Medicament"

    return None


def _candidate_values(ocr: Dict[str, Any]) -> List[Tuple[Tuple[float, str], str]]:
    raw_values = []

    if ocr.get("dosage"):
        raw_values.append(str(ocr["dosage"]))

    for value in ocr.get("dosage_candidates") or []:
        value = str(value)
        if value not in raw_values:
            raw_values.append(value)

    out = []
    seen = set()

    for raw in raw_values:
        parsed = dosage_value(raw)
        if parsed and parsed not in seen:
            seen.add(parsed)
            out.append((parsed, raw))

    return out


def _legacy_dosage_reliability(ocr: Dict[str, Any], raw: Optional[str]) -> bool:
    if not raw:
        return False

    if "dosage_reliable" in ocr:
        return bool(ocr.get("dosage_reliable"))

    votes = ocr.get("dosage_candidate_votes") or {}
    try:
        explicit_votes = int(votes.get(raw, 0))
    except Exception:
        explicit_votes = 0

    if not votes:
        return float(ocr.get("confidence", 0) or 0) >= 70

    return explicit_votes >= 3


def _select_detected_dosage(
    ocr: Dict[str, Any],
    expected: Optional[Tuple[float, str]],
) -> Tuple[Optional[Tuple[float, str]], Optional[str], bool]:
    candidates = _candidate_values(ocr)
    votes = dict(ocr.get("dosage_candidate_votes") or {})
    scores = ocr.get("dosage_candidate_scores") or {}
    number_votes = ocr.get("dosage_number_votes") or {}

    if expected and number_votes:
        family = dosage_family(expected)
        inferred_unit = {
            "mass": "mg",
            "volume": "ml",
            "percent": "%",
        }.get(family, expected[1])

        for number_text, count in number_votes.items():
            try:
                number = float(str(number_text))
                count_i = int(count)
            except Exception:
                continue

            if number <= 0 or count_i < 2:
                continue

            parsed = (round(number, 4), inferred_unit)
            raw = f"{number:g} {inferred_unit}"

            if parsed == expected:
                return parsed, raw, True

            if count_i >= 3 and dosage_family(parsed) == family:
                if (parsed, raw) not in candidates:
                    candidates.append((parsed, raw))
                votes[raw] = max(int(votes.get(raw, 0)), count_i)

    if not candidates:
        return None, None, False

    if expected:
        family = dosage_family(expected)
        candidates = [pair for pair in candidates if dosage_family(pair[0]) == family]

    if not candidates:
        return None, None, False

    def rank(pair):
        _, raw = pair
        return (int(votes.get(raw, 0)), float(scores.get(raw, 0.0)))

    parsed, raw = max(candidates, key=rank)

    # If OCR sees both the expected dosage and a nearby conflicting dosage,
    # but the conflicting read has only weak support, treat the dosage as
    # uncertain rather than creating a false WRONG_VARIANT.
    #
    # Example from a heavily blurred correct image:
    #   expected: 1000 mg
    #   OCR candidates: 1069 mg (2 weak votes), 1000 mg (1 vote)
    # A 6.9% numerical drift under severe OCR ambiguity is more plausibly a
    # recognition error than a genuine product variant.
    if expected and parsed != expected:
        expected_present = any(candidate == expected for candidate, _ in candidates)
        same_family = dosage_family(parsed) == dosage_family(expected)

        if expected_present and same_family and expected[0] > 0:
            relative_delta = abs(parsed[0] - expected[0]) / expected[0]
            top_votes = int(votes.get(raw, 0) or 0)
            top_score = float(scores.get(raw, 0.0) or 0.0)

            if (
                relative_delta <= 0.15
                and top_votes <= 2
                and top_score < 2.0
            ):
                return None, None, False

    reliable = _legacy_dosage_reliability(ocr, raw)

    if not reliable:
        return None, None, False

    return parsed, raw, True


# Words that may legitimately appear on medicine/product packaging but are not
# reliable brand identifiers by themselves.
BRAND_RESIDUAL_STOPWORDS = {
    "paracetamol", "acetaminophen", "ibuprofen", "antacid",
    "medicament", "medicine", "pharma",
    "adult", "adulte", "child", "children", "junior",
    "pain", "fever", "douleurs", "fievre",
    "oral", "voie", "tabs", "tab",
    "strength", "dosage",
}


def _segment_residual_tokens(
    meta: ProductMetadata,
    segment: str,
    known: set,
) -> List[str]:
    residual = []

    for token in tokens(segment):
        # Numeric or alphanumeric dosage/quantity fragments such as
        # "1000mg", "500mg", "10ml", "20tabs" are not brand candidates.
        if token.isdigit() or any(ch.isdigit() for ch in token) or len(token) < 3:
            continue

        if token in known:
            continue

        # Ignore likely OCR spelling variants of known product vocabulary.
        if any(
            len(k) >= 4 and _token_close(token, k)
            for k in known
        ):
            continue

        if token not in residual:
            residual.append(token)

    return residual


def _explicit_residual_brand_candidate(
    meta: ProductMetadata,
    text: str,
) -> Optional[str]:
    """Find an explicitly visible alternate brand without treating absence as mismatch.

    The OCR ensemble concatenates observations with " | ". The earliest clean
    segments normally contain the actual packaging line, while later segments
    can contain rotated/noisy OCR garbage. Therefore brand fallback is based on
    a local clean segment that contains the expected product family and dosage,
    not on every token in the giant combined OCR string.

    Example:
        DOLIPRANE 1000 mg ACME | DOLIPRANE 1000 mg ACM | ...noise...
        -> "acme"

    But:
        DOLIPRANE 1000 mg PARACETAMOL
        -> None

    Mere absence of the expected brand is never a mismatch.
    """
    if not meta.brand or not text:
        return None

    # If the expected brand is genuinely visible, no alternate-brand fallback
    # should be created.
    if exact_visible(meta.brand, text):
        return None

    known = set(GENERIC_PRODUCT_TOKENS)
    known.update(BRAND_RESIDUAL_STOPWORDS)

    for value in (
        meta.name,
        meta.category,
        meta.form,
        meta.packaging,
        meta.variant,
        meta.dosage,
        meta.strength,
    ):
        known.update(tokens(value))

    known.update(tokens(meta.brand))

    expected_dosage = _expected_dosage(meta)
    core_tokens = _core_name_tokens(meta.name)

    # Prefer the first few OCR observations. They are usually the clean
    # 0-degree recognitions; later observations often include rotated garbage.
    segments = [
        segment.strip()
        for segment in str(text).split("|")
        if segment.strip()
    ]

    for segment in segments[:8]:
        # Segment must visibly belong to the expected product family.
        segment_tokens = tokens(segment)
        if core_tokens and not all(
            any(_token_close(core, token) for token in segment_tokens)
            for core in core_tokens
        ):
            continue

        # If an expected dosage exists, require that the segment also contains
        # the same dosage. This prevents random neighbouring text from being
        # promoted to a brand.
        if expected_dosage is not None:
            segment_dosage = dosage_value(segment)
            if segment_dosage != expected_dosage:
                continue

        residual = _segment_residual_tokens(meta, segment, known)

        if len(residual) == 1:
            return residual[0]

    return None

def match_product(meta: ProductMetadata, ocr: Dict[str, Any]) -> Dict[str, Any]:
    text = str(ocr.get("text", ""))

    fields = {}
    for key in ("name", "brand", "laboratory", "form", "packaging", "variant"):
        value = getattr(meta, key, None)
        if value:
            fields[key] = fuzzy(value, text)

    core_name_score = _core_product_score(meta.name, text)
    expected_dosage = _expected_dosage(meta)
    detected_dosage, detected_dosage_raw, dosage_reliable = _select_detected_dosage(
        ocr, expected_dosage
    )

    dosage_match = None
    if expected_dosage and detected_dosage:
        dosage_match = 1.0 if expected_dosage == detected_dosage else 0.0

    detected_category = infer_category(text)
    category_match = None

    if meta.category and detected_category:
        category_match = fuzzy(meta.category, detected_category)

    ocr_reliable = bool(
        ocr.get(
            "ocr_reliable",
            ocr.get("confidence", 0) >= 50 and len(tokens(text)) >= 3,
        )
    )

    brand_score = fields.get("brand")

    # Absence of the expected brand from OCR is NOT evidence of a different
    # brand. Prefer an explicit trusted observed brand supplied by the hybrid
    # evidence resolver. If vision is unavailable, a very conservative OCR-only
    # fallback may detect one explicit residual brand token (e.g. ACME).
    observed_brand = ocr.get("observed_brand")
    observed_brand_source = "vision" if observed_brand else None

    # Core family evidence deliberately ignores dosage/form words.
    same_product_family = (
        core_name_score is None
        or core_name_score >= 0.55
        or (
            brand_score is not None
            and brand_score >= 0.70
            and category_match is not None
            and category_match >= 0.70
        )
    )

    if (
        not observed_brand
        and meta.brand
        and ocr_reliable
        and same_product_family
        and dosage_match == 1.0
        and core_name_score is not None
        and core_name_score >= 0.70
    ):
        fallback_brand = _explicit_residual_brand_candidate(meta, text)
        if fallback_brand:
            observed_brand = fallback_brand
            observed_brand_source = "ocr_residual"

    observed_brand_score = (
        fuzzy(meta.brand, str(observed_brand))
        if meta.brand and observed_brand
        else None
    )

    status = None
    reason = None

    # Category mismatch is strong only if both category signals exist.
    if category_match is not None and category_match < 0.50 and ocr_reliable:
        status = "WRONG_CATEGORY"
        reason = (
            f"Detected category '{detected_category}' conflicts with "
            f"expected category '{meta.category}'."
        )

    # Same product family + reliably different dosage is a variant mismatch.
    elif (
        dosage_match == 0.0
        and same_product_family
        and dosage_reliable
    ):
        status = "WRONG_VARIANT"
        expected_display = (
            meta.dosage
            or meta.strength
            or f"{expected_dosage[0]:g} {expected_dosage[1]}"
        )
        reason = (
            f"Visible dosage/variant '{detected_dosage_raw}' conflicts with "
            f"expected dosage '{expected_display}'."
        )

    elif (
        observed_brand_score is not None
        and observed_brand_score < 0.45
        and same_product_family
        and ocr_reliable
    ):
        status = "WRONG_BRAND"
        reason = (
            f"Detected brand '{observed_brand}' conflicts with "
            f"expected brand '{meta.brand}'."
        )

    # Do not call WRONG_PRODUCT merely because tiny/unreadable packaging text
    # failed OCR. Require reliable OCR and strong identity contradiction.
    elif (
        core_name_score is not None
        and core_name_score < 0.20
        and ocr_reliable
    ):
        status = "WRONG_PRODUCT"
        reason = "Visible product identity strongly conflicts with expected metadata."

    elif (
        fields.get("variant") is not None
        and fields["variant"] < 0.30
        and same_product_family
        and ocr_reliable
    ):
        status = "WRONG_VARIANT"
        reason = "Visible variant text conflicts with expected metadata."

    # Weighted score from evidence that is actually available.
    evidence = []

    if core_name_score is not None:
        evidence.append((core_name_score, 0.42))
    if observed_brand_score is not None:
        evidence.append((observed_brand_score, 0.18))
    elif brand_score is not None and brand_score > 0:
        # Positive OCR evidence can help, but missing brand text is neutral.
        evidence.append((brand_score, 0.18))
    if fields.get("laboratory") is not None:
        evidence.append((fields["laboratory"], 0.07))
    if fields.get("form") is not None:
        evidence.append((fields["form"], 0.07))
    if fields.get("packaging") is not None:
        evidence.append((fields["packaging"], 0.06))
    if fields.get("variant") is not None:
        evidence.append((fields["variant"], 0.06))
    if dosage_match is not None:
        evidence.append((dosage_match, 0.30))
    if category_match is not None:
        evidence.append((category_match, 0.12))

    if evidence:
        score = 100 * sum(v * w for v, w in evidence) / sum(
            w for _, w in evidence
        )
    else:
        score = 50.0

    # Uncertain OCR should lower confidence, not create a false mismatch.
    if not ocr_reliable:
        score = max(45.0, min(score, 65.0))

    verification_incomplete = bool(
        expected_dosage
        and detected_dosage is None
    )

    detected_brand = (
        str(observed_brand)
        if observed_brand
        else (
            meta.brand
            if exact_visible(meta.brand, text)
            else None
        )
    )

    detected_name = (
        meta.name
        if core_name_score is not None and core_name_score >= 0.70
        else None
    )

    detected_form = (
        meta.form
        if exact_visible(meta.form, text)
        else None
    )

    return {
        "product_match": round(score, 1),
        "mismatch_status": status,
        "reason": reason,
        "field_scores": {
            **{k: round(v * 100, 1) for k, v in fields.items()},
            "core_name": (
                None if core_name_score is None
                else round(core_name_score * 100, 1)
            ),
        },
        "expected_dosage": (
            None
            if expected_dosage is None
            else f"{expected_dosage[0]:g} {expected_dosage[1]}"
        ),
        "dosage_match": (
            None if dosage_match is None else bool(dosage_match)
        ),
        "dosage_reliable": dosage_reliable,
        "verification_incomplete": verification_incomplete,
        "detected_category": detected_category,
        "evidence_reliable": ocr_reliable,
        "observed_brand_source": observed_brand_source,
        "detected": {
            "name": detected_name,
            "brand": detected_brand,
            "category": detected_category,
            "dosage": detected_dosage_raw,
            "form": detected_form,
            "quantity": ocr.get("quantity"),
        },
    }
