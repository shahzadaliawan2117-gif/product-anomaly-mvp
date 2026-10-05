from app.models.schemas import ProductMetadata
from app.services.product_matching import match_product


def _meta():
    return ProductMetadata(
        name="Doliprane 1000 mg",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )


def _ocr(text):
    return {
        "text": text,
        "confidence": 90,
        "ocr_reliable": True,
        "dosage": "1000 mg",
        "dosage_candidates": ["1000 mg"],
        "dosage_candidate_votes": {"1000 mg": 5},
        "dosage_candidate_scores": {"1000 mg": 5.0},
        "dosage_reliable": True,
    }


def test_explicit_single_other_brand_in_ocr_is_wrong_brand():
    result = match_product(
        _meta(),
        _ocr("DOLIPRANE 1000 mg ACME"),
    )
    assert result["mismatch_status"] == "WRONG_BRAND"
    assert result["detected"]["brand"] == "acme"
    assert result["observed_brand_source"] == "ocr_residual"


def test_missing_brand_is_not_wrong_brand():
    result = match_product(
        _meta(),
        _ocr("DOLIPRANE 1000 mg"),
    )
    assert result["mismatch_status"] is None


def test_packaging_words_are_not_treated_as_other_brand():
    result = match_product(
        _meta(),
        _ocr("DOLIPRANE 1000 mg PARACETAMOL COMPRIME"),
    )
    assert result["mismatch_status"] is None
