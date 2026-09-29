import cv2, numpy as np
from typing import Dict, Any

def clamp(v): return float(max(0,min(100,v)))

def _foreground_mask(image):
    h,w=image.shape[:2]
    small=cv2.resize(image,(min(w,700), max(2,int(h*min(w,700)/w)))) if w>700 else image.copy()
    sh,sw=small.shape[:2]
    mask=np.zeros((sh,sw),np.uint8)
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    rect=(max(1,int(sw*.03)),max(1,int(sh*.03)),max(2,int(sw*.94)),max(2,int(sh*.94)))
    try:
        cv2.grabCut(small,mask,rect,bg,fg,3,cv2.GC_INIT_WITH_RECT)
        m=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    except Exception:
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY); m=(np.abs(gray.astype(float)-np.median(gray))>20).astype('uint8')
    return m

def analyze_visibility(image)->Dict[str,Any]:
    m=_foreground_mask(image); h,w=m.shape
    ys,xs=np.where(m>0)
    if len(xs)<0.01*h*w:
        return {"visibility":35.0,"bbox":None,"area_ratio":0.0,"cutoff":False,"mask":m}
    x1,x2,y1,y2=int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())
    area=((x2-x1+1)*(y2-y1+1))/(h*w)
    touches=sum([x1<=1,y1<=1,x2>=w-2,y2>=h-2])
    score=100
    if area<.18: score-=min(55,(.18-area)*300)
    if area>.92: score-=15
    score-=touches*12
    return {"visibility":round(clamp(score),1),"bbox":[x1,y1,x2,y2],"area_ratio":round(area,4),"cutoff":touches>0,"mask":m}
