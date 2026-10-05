from app.models.schemas import ProductMetadata
from app.services.product_matching import match_product


def _ocr(
    text,
    dosage=None,
    dosage_candidates=None,
    dosage_votes=None,
    number_votes=None,
    confidence=85,
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


def _meta():
    return ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )


def test_correct_1000mg_not_rejected_when_explicit_ocr_is_truncated():
    result = match_product(
        _meta(),
        _ocr(
            "Doliprane Paracetamol Comprime Sanofi",
            dosage="100 mg",
            dosage_candidates=["100 mg"],
            dosage_votes={"100 mg": 2},
            number_votes={"1000": 3},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True
    assert result["detected"]["dosage"] == "1000 mg"


def test_correct_1000mg_not_rejected_when_bad_ocr_returns_zero_fragment():
    result = match_product(
        _meta(),
        _ocr(
            "Doliprane Paracetamol Comprime Sanofi",
            dosage="00 mg",
            dosage_candidates=["00 mg"],
            dosage_votes={"00 mg": 4},
            number_votes={"1000": 2},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True


def test_real_500mg_still_detects_wrong_variant():
    result = match_product(
        _meta(),
        _ocr(
            "Doliprane Paracetamol 500 mg Comprime Sanofi",
            dosage="500 mg",
            dosage_candidates=["500 mg"],
            dosage_votes={"500 mg": 5},
            number_votes={"500": 4},
        ),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["dosage_match"] is False
    assert result["detected"]["dosage"] == "500 mg"


def test_weak_partial_100mg_without_corroboration_does_not_false_alarm():
    result = match_product(
        _meta(),
        _ocr(
            "Doliprane Paracetamol Comprime Sanofi",
            dosage="100 mg",
            dosage_candidates=["100 mg"],
            dosage_votes={"100 mg": 1},
            number_votes={},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is None
