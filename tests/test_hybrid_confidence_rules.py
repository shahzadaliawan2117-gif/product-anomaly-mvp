from app.models.schemas import ProductMetadata
from app.services.evidence_resolver import resolve_evidence
from app.services.product_matching import match_product


def _meta():
    return ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
        form="Comprime",
    )


def test_uncertain_structured_dosage_is_trusted_when_visible_text_confirms_it():
    ocr = {
        "text": "Doliprane 9 g",
        "confidence": 80,
        "dosage": "9 g",
        "dosage_reliable": True,
    }

    vision = {
        "available": True,
        "fields": {
            "product_name": "Doliprane Tabs",
            "brand": "Doliprane",
            "manufacturer": "Doliprane",
            "dosage_strength": "1000mg",
            "form": "Tabs",
            "quantity": "1000",
            "category": "Paracétamol",
            "variant": "Doliprane Tabs",
            "visible_text": [
                "Doliprane",
                "Tabs",
                "1000mg",
                "Paracétamol",
            ],
            "uncertain_fields": [
                "brand",
                "manufacturer",
                "dosage_strength",
                "form",
                "quantity",
                "category",
                "variant",
            ],
        },
    }

    evidence = resolve_evidence(ocr, vision)

    assert evidence["dosage"] == "1000mg"
    assert evidence["dosage_reliable"] is True
    assert evidence["vision_conflict"] == {
        "ocr": "9 g",
        "vision": "1000mg",
    }


def test_missing_expected_brand_in_ocr_does_not_mean_wrong_brand():
    ocr = {
        "text": "Doliprane Tabs 1000mg Paracetamol",
        "confidence": 82,
        "ocr_reliable": True,
        "dosage": "1000mg",
        "dosage_candidates": ["1000mg"],
        "dosage_candidate_votes": {"1000mg": 6},
        "dosage_candidate_scores": {"1000mg": 6.0},
        "dosage_reliable": True,
        "observed_brand": None,
        "quantity": None,
    }

    result = match_product(_meta(), ocr)

    assert result["mismatch_status"] is None
    assert result["dosage_match"] is True


def test_explicit_trusted_different_brand_can_still_be_wrong_brand():
    meta = ProductMetadata(
        name="Example Product 1000 mg",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
    )

    ocr = {
        "text": "Example Product 1000 mg Pfizer",
        "confidence": 90,
        "ocr_reliable": True,
        "dosage": "1000 mg",
        "dosage_candidates": ["1000 mg"],
        "dosage_candidate_votes": {"1000 mg": 5},
        "dosage_candidate_scores": {"1000 mg": 5.0},
        "dosage_reliable": True,
        "observed_brand": "Pfizer",
    }

    result = match_product(meta, ocr)

    assert result["mismatch_status"] == "WRONG_BRAND"
