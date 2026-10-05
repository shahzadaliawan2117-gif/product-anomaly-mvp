from app.models.schemas import ProductMetadata, AnalyzeResponse, Scores, DetectedProduct
from app.services.ocr import extract_ocr
from app.services.product_matching import match_product
from app.services.image_quality import analyze_quality
from app.services.visibility import analyze_visibility
from app.services.composition import analyze_composition
from app.services.background import analyze_background
from app.services.scoring import overall_score
from app.services.classifier import classify
from app.services.visual_similarity import compare_to_reference
from app.services.vision_verifier import inspect_image
from app.services.evidence_resolver import resolve_evidence


def _trusted_vision_field(vision, key):
    if not vision.get("available"):
        return None

    fields = vision.get("fields") or {}
    uncertain = set(fields.get("uncertain_fields") or [])

    if key in uncertain:
        return None

    value = fields.get(key)
    return str(value).strip() if value else None


def analyze(image, meta: ProductMetadata) -> AnalyzeResponse:
    # 1) Independent observations. Expected metadata is NOT sent to the VLM.
    ocr = extract_ocr(image)
    vision = inspect_image(image)

    # 2) Fuse trustworthy OCR + local visual evidence into the format already
    # understood by the deterministic matcher.
    evidence = resolve_evidence(ocr, vision)
    match = match_product(meta, evidence)

    # 3) Optional reference-image similarity remains supplementary evidence.
    visual = compare_to_reference(
        image,
        meta.model_dump(mode="python"),
    )
    if visual.get("available"):
        # Never let visual similarity override a confirmed variant/category
        # mismatch.
        match["product_match"] = round(
            0.80 * match["product_match"]
            + 0.20 * visual["similarity"],
            1,
        )

    # 4) Objective image-quality pipeline remains deterministic/OpenCV based.
    q = analyze_quality(image)
    vis = analyze_visibility(image)
    comp = analyze_composition(image, vis)
    bg = analyze_background(image, vis)

    text_readability = max(
        0,
        min(100, float(ocr.get("confidence", 0) or 0) * 1.15),
    )

    score_dict = {
        "product_match": match["product_match"],
        "image_quality": q["image_quality"],
        "sharpness": q["sharpness"],
        "resolution": q["resolution"],
        "visibility": vis["visibility"],
        "composition": comp["composition"],
        "background": bg["background"],
        "lighting": q["lighting"],
        "text_readability": round(text_readability, 1),
    }
    score_dict["overall"] = overall_score(score_dict)

    status, is_anomaly, anomaly_types = classify(
        match,
        score_dict,
        q["issues"],
        [],
        comp["issues"],
        bg["issues"],
    )

    correct = not status.startswith("WRONG_")

    issues = []
    if match.get("reason"):
        issues.append(match["reason"])

    issues += [
        x.replace("_", " ").title()
        for x in anomaly_types
        if x not in {
            "WRONG_PRODUCT",
            "WRONG_BRAND",
            "WRONG_VARIANT",
            "WRONG_CATEGORY",
        }
    ]

    if not ocr.get("text") and not vision.get("available"):
        issues.append(
            "Packaging text could not be read reliably; "
            "product-match confidence is limited."
        )

    if vision.get("enabled") and not vision.get("available"):
        issues.append(
            "Local visual verifier was unavailable; "
            "analysis fell back to OCR/CV evidence."
        )

    if correct and is_anomaly:
        explanation = (
            "The image appears consistent with the expected product, "
            "but one or more visual-quality or presentation issues were detected."
        )
    elif correct:
        explanation = (
            "The image appears consistent with the provided product metadata "
            "and meets the configured visual-quality thresholds."
        )
    else:
        explanation = (
            match.get("reason")
            or "The image does not sufficiently match the provided product metadata."
        )

    # 5) DetectedProduct must describe the IMAGE, not echo expected metadata.
    det = dict(match["detected"])

    vision_name = _trusted_vision_field(vision, "product_name")
    vision_brand = _trusted_vision_field(vision, "brand")
    vision_category = _trusted_vision_field(vision, "category")
    vision_form = _trusted_vision_field(vision, "form")
    vision_quantity = _trusted_vision_field(vision, "quantity")

    if vision_name:
        det["name"] = vision_name
    if vision_brand:
        det["brand"] = vision_brand
    if vision_category:
        det["category"] = vision_category
    if vision_form:
        det["form"] = vision_form
    if vision_quantity:
        det["quantity"] = vision_quantity

    # Use only the dosage selected by the hybrid resolver/matcher.
    if evidence.get("dosage_reliable") and evidence.get("dosage"):
        det["dosage"] = evidence["dosage"]

    return AnalyzeResponse(
        product_id=meta.id,
        is_correct_product=correct,
        is_anomaly=is_anomaly,
        status=status,
        anomaly_types=anomaly_types,
        scores=Scores(**score_dict),
        detected_product=DetectedProduct(**det),
        issues=list(dict.fromkeys(issues)),
        explanation=explanation,
        diagnostics={
            "ocr": {
                "text": ocr.get("text", ""),
                "confidence": ocr.get("confidence", 0),
                "dosage": ocr.get("dosage"),
                "dosage_reliable": ocr.get("dosage_reliable", False),
            },
            "local_vision": vision,
            "hybrid_evidence": {
                "source": evidence.get("hybrid_source"),
                "dosage": evidence.get("dosage"),
                "dosage_reliable": evidence.get("dosage_reliable", False),
                "vision_conflict": evidence.get("vision_conflict"),
            },
            "match": {
                k: v
                for k, v in match.items()
                if k not in {"detected"}
            },
            "quality": q["metrics"],
            "visibility": {
                k: v
                for k, v in vis.items()
                if k != "mask"
            },
            "composition": comp,
            "background": bg,
            "visual_similarity": visual,
        },
    )
