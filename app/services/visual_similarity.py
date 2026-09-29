"""Deterministic visual descriptor + cosine similarity for optional reference images.

The API still has only two conceptual inputs: image + product_json. If product_json
contains `reference_image_base64` or `reference_image_url`, this module adds a
non-OCR visual-similarity signal. It is intentionally replaceable by CLIP/SigLIP.
"""
from typing import Optional, Dict, Any
import cv2, numpy as np
from app.utils.image_loader import from_base64, from_url

def _descriptor(image: np.ndarray) -> np.ndarray:
    img=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]).flatten().astype(np.float32)
    hist/=np.linalg.norm(hist)+1e-8
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,70,150)
    # Spatial edge-density grid captures coarse package geometry/layout.
    grid=[]
    for y in range(0,256,64):
        for x in range(0,256,64):
            grid.append(float(np.mean(edges[y:y+64,x:x+64]>0)))
    v=np.concatenate([hist,np.asarray(grid,dtype=np.float32)])
    return v/(np.linalg.norm(v)+1e-8)

def compare_to_reference(image: np.ndarray, metadata_dict: Dict[str,Any]) -> Dict[str,Any]:
    ref=None
    try:
        if metadata_dict.get('reference_image_base64'):
            ref=from_base64(metadata_dict['reference_image_base64'])
        elif metadata_dict.get('reference_image_url'):
            ref=from_url(metadata_dict['reference_image_url'])
    except Exception as e:
        return {'available':False,'similarity':None,'error':f'Reference image unavailable: {e}'}
    if ref is None:
        return {'available':False,'similarity':None,'error':None}
    a,b=_descriptor(image),_descriptor(ref)
    sim=float(np.dot(a,b))
    return {'available':True,'similarity':round(max(0,min(1,sim))*100,1),'error':None}
