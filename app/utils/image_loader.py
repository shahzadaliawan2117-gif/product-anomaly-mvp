import base64, io
from urllib.parse import urlparse
import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from fastapi import HTTPException
from app.config import settings

ALLOWED = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

def _decode_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "Empty image data")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, "Image exceeds maximum allowed size")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED:
                raise HTTPException(415, f"Unsupported image format: {im.format}")
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(400, f"Invalid or corrupted image: {e}")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def from_url(url: str) -> np.ndarray:
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise HTTPException(400, "Image URL must use http or https")
    try:
        r = requests.get(url, timeout=settings.request_timeout_seconds, stream=True)
        r.raise_for_status()
        data = r.content
    except requests.RequestException as e:
        raise HTTPException(400, f"Image URL unavailable: {e}")
    return _decode_bytes(data)

def from_base64(value: str) -> np.ndarray:
    if value.startswith("data:"):
        try: value = value.split(",", 1)[1]
        except IndexError: raise HTTPException(400, "Invalid data URL")
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")
    return _decode_bytes(data)

def from_bytes(data: bytes) -> np.ndarray:
    return _decode_bytes(data)
