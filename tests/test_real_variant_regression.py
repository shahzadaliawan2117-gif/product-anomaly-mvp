from app.models.schemas import ProductMetadata
from app.services.product_matching import match_product


def _ocr(text, dosage=None, confidence=85):
    return {
        "text": text,
        "confidence": confidence,
        "dosage": dosage,
        "dosage_candidates": [] if dosage is None else [dosage],
        "dosage_candidate_votes": {} if dosage is None else {dosage: 4},
        "quantity": None,
        "error": None,
    }


def test_doliprane_vitamin_c_500_vs_doliprane_1000_is_wrong_variant():
    meta = ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        laboratory="Sanofi",
        dosage="1000 mg",
        form="Comprime",
        packaging="Boite de 8 comprimes",
    )

    result = match_product(
        meta,
        _ocr(
            "Paracetamol Doliprane Vitamine C 500 mg 150 mg "
            "comprimés effervescents Sanofi",
            dosage="500 mg",
        ),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"
    assert result["dosage_match"] is False


def test_low_full_name_score_with_same_brand_category_and_dosage_mismatch_is_variant():
    meta = ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
    )

    # Simulates weak OCR of the product name but reliable brand/category/dosage.
    result = match_product(
        meta,
        _ocr("Sanofi Paracetamol 500 mg medicine", dosage="500 mg"),
    )

    assert result["mismatch_status"] == "WRONG_VARIANT"


def test_unrelated_product_is_still_wrong_product():
    meta = ProductMetadata(
        name="Doliprane 1000 mg Comprime",
        category="Medicament",
        brand="Sanofi",
        dosage="1000 mg",
    )

    result = match_product(
        meta,
        _ocr("Maalox antacid oral suspension 200 ml", dosage="200 ml"),
    )

    assert result["mismatch_status"] == "WRONG_PRODUCT"
