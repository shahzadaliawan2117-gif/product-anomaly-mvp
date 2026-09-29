from pydantic import BaseModel, Field
from typing import Dict

class Settings(BaseModel):
    max_image_bytes: int = 15 * 1024 * 1024
    request_timeout_seconds: int = 10
    min_resolution_side: int = 480
    target_resolution_side: int = 1200
    blur_bad_laplacian: float = 45.0
    blur_good_laplacian: float = 220.0
    low_quality_threshold: float = 55.0
    valid_threshold: float = 70.0
    product_match_threshold: float = 58.0
    strong_mismatch_threshold: float = 35.0
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "image_quality": 0.20,
        "sharpness": 0.14,
        "resolution": 0.10,
        "visibility": 0.14,
        "composition": 0.10,
        "background": 0.08,
        "lighting": 0.10,
        "text_readability": 0.14,
    })

settings = Settings()
