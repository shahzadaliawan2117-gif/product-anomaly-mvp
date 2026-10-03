from app.models.schemas import ProductMetadata
from app.services.ocr import _extract_dosage_candidates
from app.services.product_matching import match_product


def _ocr(text, dosage=None, candidates=None, votes=None, confidence=85):
    return {
        "text": text,
        "confidence": confidence,
        "dosage": dosage,
        "dosage_candidates": candidates or ([] if dosage is None else [dosage]),
        "dosage_candidate_votes": votes or ({} if dosage is None else {dosage: 3}),
        "quantity": None,
        "error": None,
    }


def test_ocr_recovers_500mg_from_common_letter_confusion():
    candidates = _extract_dosage_candidates("Doliprane SOO mg Comprime")
    assert "500 mg" in candidates


def test_ocr_recovers_1000mg_without_space():
    candidates = _extract_dosage_candidates("Doliprane 1000mg Comprime")
    assert "1000 mg" in candidates


def test_500mg_image_vs_1000mg_json_is_wrong_variant():
    meta = ProductMetadata(
        id="12345",
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )

    result = match_product(
        meta,
        _ocr(
            "Doliprane Paracetamol 500 mg Comprime Sanofi",
            dosage="500 mg",
            candidates=["500 mg"],
            votes={"500 mg": 4},
        ),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["dosage_match"] is False
    assert result["expected_dosage"] == "1000 mg"


def test_strength_is_inferred_from_product_name_when_dosage_field_missing():
    meta = ProductMetadata(
        id="12345",
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        form="Comprime",
    )

    result = match_product(
        meta,
        _ocr(
            "Doliprane Paracetamol 500 mg Comprime Sanofi",
            dosage="500 mg",
            candidates=["500 mg"],
            votes={"500 mg": 3},
        ),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["expected_dosage"] == "1000 mg"


def test_same_1000mg_variant_remains_valid_from_matching_layer():
    meta = ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )

    result = match_product(
        meta,
        _ocr(
            "Doliprane Paracetamol 1000 mg Comprime Sanofi",
            dosage="1000 mg",
            candidates=["1000 mg"],
            votes={"1000 mg": 4},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True


def test_1g_and_1000mg_are_same_strength():
    meta = ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
    )

    result = match_product(
        meta,
        _ocr(
            "Doliprane Paracetamol 1 g Comprime Sanofi",
            dosage="1 g",
            candidates=["1 g"],
            votes={"1 g": 3},
        ),
    )

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True


def test_package_volume_does_not_conflict_with_mass_strength():
    meta = ProductMetadata(
        name="Example Medicine 1000 mg",
        category="Medicament",
        dosage="1000 mg",
    )

    result = match_product(
        meta,
        _ocr(
            "Example Medicine 200 ml bottle",
            dosage="200 ml",
            candidates=["200 ml"],
            votes={"200 ml": 4},
        ),
    )

    assert result["dosage_match"] is None
