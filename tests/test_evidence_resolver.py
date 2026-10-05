from app.services.evidence_resolver import resolve_evidence


def test_vision_200mg_overrides_conflicting_ocr_4g():
    ocr = {
        "text": "Doliprane 4 g",
        "confidence": 70,
        "dosage": "4 g",
        "dosage_candidates": ["4 g"],
        "dosage_candidate_votes": {"4 g": 4},
        "dosage_candidate_scores": {"4 g": 3.2},
        "dosage_reliable": True,
        "quantity": None,
    }

    vision = {
        "available": True,
        "fields": {
            "product_name": "Doliprane",
            "brand": "Sanofi",
            "manufacturer": "Sanofi",
            "dosage_strength": "200 mg",
            "form": "tablette",
            "quantity": "10",
            "category": "paracetamol",
            "variant": "Doliprane 200mg",
            "visible_text": [
                "Doliprane 200mg",
                "douleurs et fièvre",
            ],
            "uncertain_fields": [],
        },
    }

    evidence = resolve_evidence(ocr, vision)

    assert evidence["dosage"] == "200 mg"
    assert evidence["dosage_reliable"] is True
    assert evidence["vision_conflict"] == {
        "ocr": "4 g",
        "vision": "200 mg",
    }


def test_unsubstantiated_vision_dosage_does_not_override_ocr():
    ocr = {
        "text": "Doliprane 500 mg",
        "confidence": 80,
        "dosage": "500 mg",
        "dosage_candidates": ["500 mg"],
        "dosage_candidate_votes": {"500 mg": 4},
        "dosage_candidate_scores": {"500 mg": 3.0},
        "dosage_reliable": True,
    }

    vision = {
        "available": True,
        "fields": {
            "product_name": "Doliprane",
            "brand": "Sanofi",
            "dosage_strength": "200 mg",
            "visible_text": ["Doliprane"],
            "uncertain_fields": [],
        },
    }

    evidence = resolve_evidence(ocr, vision)

    assert evidence["dosage"] == "500 mg"


def test_disabled_vision_preserves_ocr_evidence():
    ocr = {
        "text": "Doliprane 500 mg",
        "dosage": "500 mg",
        "dosage_reliable": True,
    }

    evidence = resolve_evidence(
        ocr,
        {
            "available": False,
            "enabled": False,
            "fields": {},
            "error": None,
        },
    )

    assert evidence["dosage"] == "500 mg"
    assert evidence["hybrid_source"] == "ocr_only"
