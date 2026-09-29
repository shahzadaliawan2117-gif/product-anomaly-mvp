from app.config import settings

def overall_score(scores:dict)->float:
    vals=[]; weights=[]
    for k,w in settings.weights.items():
        if scores.get(k) is not None:
            vals.append(float(scores[k])*w); weights.append(w)
    return round(sum(vals)/sum(weights),1) if weights else 0.0
