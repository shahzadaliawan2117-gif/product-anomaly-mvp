from typing import Dict, Any, List
import cv2, numpy as np
from app.config import settings

def clamp(v): return float(max(0,min(100,v)))
def lin_score(v, bad, good):
    if v <= bad: return 0.0
    if v >= good: return 100.0
    return 100*(v-bad)/(good-bad)

def analyze_quality(image: np.ndarray) -> Dict[str, Any]:
    h,w=image.shape[:2]; gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    sharp=lin_score(lap, settings.blur_bad_laplacian, settings.blur_good_laplacian)
    min_side=min(h,w)
    resolution=clamp(100*min_side/settings.target_resolution_side)
    mean=float(gray.mean()); std=float(gray.std())
    dark_ratio=float(np.mean(gray<25)); bright_ratio=float(np.mean(gray>245))
    # White catalog backgrounds are valid, so clipping is severe only when it dominates nearly the entire frame.
    exposure_penalty=max(0,(dark_ratio-0.75)*300)+max(0,(bright_ratio-0.85)*300)
    if std < 18:
        exposure_penalty += max(0,55-mean)*1.2 + max(0,mean-242)*1.2
    lighting=clamp(100-exposure_penalty)
    contrast=clamp((std/64)*100)
    # Robust high-frequency residual noise estimate.
    den=cv2.GaussianBlur(gray,(3,3),0)
    noise=float(np.median(np.abs(gray.astype(np.float32)-den.astype(np.float32))))
    noise_score=clamp(100-noise*6)
    # Block boundary discontinuity: heuristic for heavy JPEG/block artifacts.
    f=gray.astype(np.float32)
    vb=np.mean(np.abs(f[:,8::8]-f[:,7:-1:8])) if w>16 else 0
    hb=np.mean(np.abs(f[8::8,:]-f[7:-1:8,:])) if h>16 else 0
    blockiness=float((vb+hb)/2)
    compression_score=clamp(100-max(0,blockiness-8)*4)
    quality=0.27*sharp+0.18*resolution+0.20*lighting+0.13*contrast+0.12*noise_score+0.10*compression_score
    issues: List[str]=[]
    if sharp<45: issues.append("BLUR")
    if sharp<20: issues.append("EXCESSIVE_BLUR")
    if resolution<45: issues.append("LOW_RESOLUTION")
    if noise_score<45: issues.append("EXCESSIVE_NOISE")
    if compression_score<45: issues.append("COMPRESSION_ARTIFACTS")
    if bright_ratio>0.90 and std<25: issues.append("OVEREXPOSURE")
    if dark_ratio>0.80 and std<25: issues.append("UNDEREXPOSURE")
    if contrast<38: issues.append("POOR_CONTRAST")
    return {"image_quality":round(quality,1),"sharpness":round(sharp,1),"resolution":round(resolution,1),"lighting":round(lighting,1),
      "metrics":{"width":w,"height":h,"laplacian_variance":round(lap,2),"mean_brightness":round(mean,2),"contrast_std":round(std,2),"dark_ratio":round(dark_ratio,4),"bright_ratio":round(bright_ratio,4),"noise_median_residual":round(noise,2),"blockiness":round(blockiness,2)},"issues":issues}
