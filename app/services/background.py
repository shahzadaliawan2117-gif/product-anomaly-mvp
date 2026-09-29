import cv2, numpy as np
from typing import Dict, Any

def clamp(v): return float(max(0,min(100,v)))
def analyze_background(image, visibility: Dict[str,Any])->Dict[str,Any]:
    m=visibility.get('mask')
    if m is None: return {"background":50.0,"issues":["BACKGROUND_UNCERTAIN"]}
    if m.shape[:2] != image.shape[:2]:
        img=cv2.resize(image,(m.shape[1],m.shape[0]))
    else: img=image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bg=gray[m==0]
    if bg.size<100: return {"background":55.0,"issues":["BACKGROUND_NOT_MEASURABLE"]}
    std=float(bg.std())
    edges=cv2.Canny(gray,80,160)
    edge_density=float(np.mean(edges[m==0]>0)) if np.any(m==0) else 0
    # Smooth/low-edge catalog-like background scores higher; not mandatory white.
    score=clamp(100 - max(0,std-24)*1.0 - max(0,edge_density-.035)*320)
    issues=[]
    if score<50: issues.append("DISTRACTING_BACKGROUND")
    return {"background":round(score,1),"issues":issues,"background_std":round(std,2),"background_edge_density":round(edge_density,4)}
