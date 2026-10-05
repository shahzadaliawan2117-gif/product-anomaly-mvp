from app.models.schemas import ProductMetadata
from app.services.product_matching import match_product


def meta(dosage="1000 mg"):
    return ProductMetadata(
        name=f"Doliprane {dosage} Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage=dosage,
        form="Comprime",
    )


def ocr(
    text,
    dosage=None,
    reliable=True,
    ocr_reliable=True,
    votes=None,
    scores=None,
):
    return {
        "text": text,
        "confidence": 80 if ocr_reliable else 35,
        "ocr_reliable": ocr_reliable,
        "dosage": dosage,
        "dosage_candidates": [] if dosage is None else [dosage],
        "dosage_candidate_votes": votes or ({} if dosage is None else {dosage: 4}),
        "dosage_candidate_scores": scores or ({} if dosage is None else {dosage: 3.0}),
        "dosage_reliable": reliable if dosage is not None else False,
        "dosage_number_votes": {},
        "quantity": None,
        "error": None,
    }


def test_reliable_500_vs_1000_is_wrong_variant():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "Doliprane Paracetamol 500 mg Sanofi",
            dosage="500 mg",
        ),
    )
    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["dosage_match"] is False


def test_reliable_200_vs_1000_is_wrong_variant():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "Doliprane Paracetamol 200 mg Sanofi",
            dosage="200 mg",
        ),
    )
    assert result["mismatch_status"] == "WRONG_VARIANT"


def test_1g_equals_1000mg():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "Doliprane Paracetamol 1 g Sanofi",
            dosage="1 g",
        ),
    )
    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True


def test_single_unreliable_9g_does_not_create_false_mismatch():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "noisy tiny packaging text",
            dosage="9 g",
            reliable=False,
            ocr_reliable=False,
            votes={"9 g": 1},
            scores={"9 g": 0.5},
        ),
    )
    assert result["mismatch_status"] is None
    assert result["dosage_match"] is None
    assert result["detected"]["dosage"] is None


def test_unreadable_text_does_not_become_wrong_product():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "garbled xzq rrr text",
            dosage=None,
            ocr_reliable=False,
        ),
    )
    assert result["mismatch_status"] is None
    assert result["verification_incomplete"] is True


def test_high_confidence_other_product_is_wrong_product():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "Maalox Antacid oral suspension 200 ml",
            dosage="200 ml",
            reliable=True,
            ocr_reliable=True,
        ),
    )
    assert result["mismatch_status"] == "WRONG_PRODUCT"


def test_shampoo_is_wrong_category():
    result = match_product(
        meta("1000 mg"),
        ocr(
            "Pantene shampoo hair conditioner",
            dosage=None,
            ocr_reliable=True,
        ),
    )
    assert result["mismatch_status"] == "WRONG_CATEGORY"
