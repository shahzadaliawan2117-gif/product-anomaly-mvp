from app.models.schemas import ProductMetadata
from app.services.product_matching import match_product


def _meta():
    return ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )


def _ocr(
    text,
    confidence,
    dosage=None,
    dosage_candidates=None,
    dosage_votes=None,
    number_votes=None,
):
    return {
        "text": text,
        "confidence": confidence,
        "dosage": dosage,
        "dosage_candidates": dosage_candidates or ([] if dosage is None else [dosage]),
        "dosage_candidate_votes": dosage_votes or {},
        "dosage_number_votes": number_votes or {},
        "quantity": None,
        "error": None,
    }


def test_weak_9g_read_does_not_become_wrong_product_or_wrong_variant():
    result = match_product(
        _meta(),
        _ocr(
            text="garbled low resolution packaging text",
            confidence=42.6,
            dosage="9 g",
            dosage_candidates=["9 g"],
            dosage_votes={"9 g": 1},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is None
    assert result["detected"]["dosage"] is None


def test_weak_zero_fragment_is_not_displayed_as_detected_dosage():
    result = match_product(
        _meta(),
        _ocr(
            text="Doliprane paracetamol",
            confidence=42.0,
            dosage="00 mg",
            dosage_candidates=["00 mg"],
            dosage_votes={"00 mg": 2},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["detected"]["dosage"] is None


def test_strong_repeated_500mg_remains_wrong_variant():
    result = match_product(
        _meta(),
        _ocr(
            text="Doliprane Paracetamol 500 mg Sanofi",
            confidence=40.0,
            dosage="500 mg",
            dosage_candidates=["500 mg"],
            dosage_votes={"500 mg": 5},
            number_votes={"500": 4},
        ),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["dosage_match"] is False
    assert result["detected"]["dosage"] == "500 mg"


def test_high_confidence_unrelated_product_can_still_be_wrong_product():
    result = match_product(
        _meta(),
        _ocr(
            text="Maalox antacid oral suspension",
            confidence=88.0,
        ),
    )

    assert result["mismatch_status"] == "WRONG_PRODUCT"
