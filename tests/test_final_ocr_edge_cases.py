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


def test_clean_acme_segment_is_detected_despite_later_ocr_noise():
    ocr = {
        "text": (
            "DOLIPRANE 1000 mg ACME | "
            "DOLIPRANE 1000 mg ACM | "
            "0 fe Cc m random rotated OCR garbage"
        ),
        "confidence": 96.0,
        "ocr_reliable": True,
        "dosage": "1000 mg",
        "dosage_candidates": ["1000 mg"],
        "dosage_candidate_votes": {"1000 mg": 10},
        "dosage_candidate_scores": {"1000 mg": 10.3},
        "dosage_reliable": True,
    }

    result = match_product(_meta(), ocr)

    assert result["mismatch_status"] == "WRONG_BRAND"
    assert result["detected"]["brand"] == "acme"


def test_nearby_weak_blur_dosage_conflict_becomes_uncertain_not_wrong_variant():
    ocr = {
        "text": "DOLIPRANE 1069 mg | DOLIPRANE 1000 mg SAN",
        "confidence": 53.08,
        "ocr_reliable": True,
        "dosage": "1069 mg",
        "dosage_candidates": ["1069 mg", "1000 mg"],
        "dosage_candidate_votes": {
            "1069 mg": 2,
            "1000 mg": 1,
        },
        "dosage_candidate_scores": {
            "1069 mg": 1.345,
            "1000 mg": 0.656,
        },
        "dosage_reliable": True,
    }

    result = match_product(_meta(), ocr)

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is None
    assert result["detected"]["dosage"] is None


def test_clear_500mg_still_wrong_variant():
    ocr = {
        "text": "DOLIPRANE 500 mg SANOFI",
        "confidence": 95,
        "ocr_reliable": True,
        "dosage": "500 mg",
        "dosage_candidates": ["500 mg"],
        "dosage_candidate_votes": {"500 mg": 6},
        "dosage_candidate_scores": {"500 mg": 6.0},
        "dosage_reliable": True,
    }

    result = match_product(_meta(), ocr)

    assert result["mismatch_status"] == "WRONG_VARIANT"
