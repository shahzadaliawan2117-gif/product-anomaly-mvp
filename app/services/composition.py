from typing import Dict, Any

def clamp(v): return float(max(0,min(100,v)))
def analyze_composition(image, visibility: Dict[str,Any])->Dict[str,Any]:
    h,w=image.shape[:2]; bbox=visibility.get('bbox')
    if not bbox: return {"composition":35.0,"issues":["PRODUCT_NOT_CLEarly_LOCALIZED"]}
    x1,y1,x2,y2=bbox; cx=(x1+x2)/2/w; cy=(y1+y2)/2/h; area=visibility['area_ratio']
    center_dist=((cx-.5)**2+(cy-.5)**2)**0.5
    size_pen=0 if .28<=area<=.80 else min(35, abs(area-.54)*80)
    center_pen=min(30,center_dist*75)
    edge_margin=min(x1/w,y1/h,(w-1-x2)/w,(h-1-y2)/h)
    edge_pen=max(0,(.03-edge_margin)*500)
    score=clamp(100-size_pen-center_pen-edge_pen)
    issues=[]
    if area<.18: issues.append("PRODUCT_TOO_SMALL")
    if area>.92: issues.append("FRAMING_TOO_TIGHT")
    if center_dist>.30: issues.append("POOR_POSITIONING")
    return {"composition":round(score,1),"issues":issues,"center":[round(cx,3),round(cy,3)],"edge_margin":round(edge_margin,4)}
