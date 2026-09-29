from app.config import settings

def classify(match, scores, quality_issues, vis_issues, comp_issues, bg_issues):
    mismatch=match.get('mismatch_status')
    all_issues=list(dict.fromkeys(quality_issues+vis_issues+comp_issues+bg_issues))
    if mismatch:
        return mismatch, True, [mismatch]+all_issues
    severe=[x for x in all_issues if x in {'EXCESSIVE_BLUR','LOW_RESOLUTION','UNDEREXPOSURE','OVEREXPOSURE','DISTRACTING_BACKGROUND','PRODUCT_TOO_SMALL'}]
    if scores['product_match'] < settings.strong_mismatch_threshold:
        return 'WRONG_PRODUCT', True, ['WRONG_PRODUCT']+all_issues
    if scores['overall'] < settings.low_quality_threshold or severe:
        return 'VALID_BUT_LOW_QUALITY', True, all_issues or ['LOW_IMAGE_QUALITY']
    if all_issues and scores['overall'] < settings.valid_threshold:
        return 'ANOMALY', True, all_issues
    return 'VALID', False, all_issues
