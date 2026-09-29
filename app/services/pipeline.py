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

def analyze(image, meta: ProductMetadata)->AnalyzeResponse:
    ocr=extract_ocr(image)
    match=match_product(meta,ocr)
    visual=compare_to_reference(image, meta.model_dump(mode='python'))
    if visual.get('available'):
        # Supplement textual evidence; do not let visual similarity override a confirmed variant/category mismatch.
        match['product_match']=round(0.80*match['product_match']+0.20*visual['similarity'],1)
    q=analyze_quality(image)
    vis=analyze_visibility(image)
    comp=analyze_composition(image,vis)
    bg=analyze_background(image,vis)
    text_readability=max(0,min(100, ocr.get('confidence',0)*1.15))
    score_dict={"product_match":match['product_match'],"image_quality":q['image_quality'],"sharpness":q['sharpness'],"resolution":q['resolution'],"visibility":vis['visibility'],"composition":comp['composition'],"background":bg['background'],"lighting":q['lighting'],"text_readability":round(text_readability,1)}
    score_dict['overall']=overall_score(score_dict)
    status,is_anomaly,anomaly_types=classify(match,score_dict,q['issues'],[],comp['issues'],bg['issues'])
    correct=not status.startswith('WRONG_')
    issues=[]
    if match.get('reason'): issues.append(match['reason'])
    issues += [x.replace('_',' ').title() for x in anomaly_types if x not in {'WRONG_PRODUCT','WRONG_BRAND','WRONG_VARIANT','WRONG_CATEGORY'}]
    if not ocr.get('text'):
        issues.append('Packaging text could not be read reliably; product-match confidence is limited.')
    if correct and is_anomaly:
        explanation='The image appears consistent with the expected product, but one or more visual-quality or presentation issues were detected.'
    elif correct:
        explanation='The image appears consistent with the provided product metadata and meets the configured visual-quality thresholds.'
    else:
        explanation=match.get('reason') or 'The image does not sufficiently match the provided product metadata.'
    det=match['detected']
    return AnalyzeResponse(product_id=meta.id,is_correct_product=correct,is_anomaly=is_anomaly,status=status,anomaly_types=anomaly_types,
      scores=Scores(**score_dict),detected_product=DetectedProduct(**det),issues=list(dict.fromkeys(issues)),explanation=explanation,
      diagnostics={"ocr":{"text":ocr.get('text',''),"confidence":ocr.get('confidence',0)},"match":{k:v for k,v in match.items() if k not in {'detected'}},"quality":q['metrics'],"visibility":{k:v for k,v in vis.items() if k!='mask'},"composition":comp,"background":bg,"visual_similarity":visual})
