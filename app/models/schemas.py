from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class ProductMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    laboratory: Optional[str] = None
    dosage: Optional[str] = None
    strength: Optional[str] = None
    form: Optional[str] = None
    packaging: Optional[str] = None
    quantity: Optional[str] = None
    variant: Optional[str] = None

class DetectedProduct(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    dosage: Optional[str] = None
    form: Optional[str] = None
    quantity: Optional[str] = None

class Scores(BaseModel):
    product_match: float
    image_quality: Optional[float] = None
    sharpness: Optional[float] = None
    resolution: Optional[float] = None
    visibility: Optional[float] = None
    composition: Optional[float] = None
    background: Optional[float] = None
    lighting: Optional[float] = None
    text_readability: Optional[float] = None
    overall: Optional[float] = None

class AnalyzeResponse(BaseModel):
    product_id: Optional[str] = None
    is_correct_product: bool
    is_anomaly: bool
    status: str
    anomaly_types: List[str] = Field(default_factory=list)
    scores: Scores
    detected_product: DetectedProduct
    issues: List[str] = Field(default_factory=list)
    explanation: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
