import os
from typing import Dict

from pydantic import BaseModel, Field, model_validator


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


class Settings(BaseModel):
    max_image_bytes: int = Field(
        default_factory=lambda: _env_int("MAX_IMAGE_BYTES", 15 * 1024 * 1024)
    )
    request_timeout_seconds: int = Field(
        default_factory=lambda: _env_int("REQUEST_TIMEOUT_SECONDS", 10)
    )

    min_resolution_side: int = Field(
        default_factory=lambda: _env_int("MIN_RESOLUTION_SIDE", 480)
    )
    target_resolution_side: int = Field(
        default_factory=lambda: _env_int("TARGET_RESOLUTION_SIDE", 1200)
    )

    blur_bad_laplacian: float = Field(
        default_factory=lambda: _env_float("BLUR_BAD_LAPLACIAN", 45.0)
    )
    blur_good_laplacian: float = Field(
        default_factory=lambda: _env_float("BLUR_GOOD_LAPLACIAN", 220.0)
    )

    low_quality_threshold: float = Field(
        default_factory=lambda: _env_float("LOW_QUALITY_THRESHOLD", 55.0)
    )
    valid_threshold: float = Field(
        default_factory=lambda: _env_float("VALID_THRESHOLD", 70.0)
    )
    product_match_threshold: float = Field(
        default_factory=lambda: _env_float("PRODUCT_MATCH_THRESHOLD", 58.0)
    )
    strong_mismatch_threshold: float = Field(
        default_factory=lambda: _env_float("STRONG_MISMATCH_THRESHOLD", 35.0)
    )

    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "image_quality": _env_float("WEIGHT_IMAGE_QUALITY", 0.20),
            "sharpness": _env_float("WEIGHT_SHARPNESS", 0.14),
            "resolution": _env_float("WEIGHT_RESOLUTION", 0.10),
            "visibility": _env_float("WEIGHT_VISIBILITY", 0.14),
            "composition": _env_float("WEIGHT_COMPOSITION", 0.10),
            "background": _env_float("WEIGHT_BACKGROUND", 0.08),
            "lighting": _env_float("WEIGHT_LIGHTING", 0.10),
            "text_readability": _env_float("WEIGHT_TEXT_READABILITY", 0.14),
        }
    )

    @model_validator(mode="after")
    def validate_settings(self):
        if self.max_image_bytes <= 0:
            raise ValueError("MAX_IMAGE_BYTES must be greater than 0")

        if self.request_timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than 0")

        if self.min_resolution_side <= 0 or self.target_resolution_side <= 0:
            raise ValueError("Resolution settings must be greater than 0")

        if self.target_resolution_side < self.min_resolution_side:
            raise ValueError(
                "TARGET_RESOLUTION_SIDE must be greater than or equal to MIN_RESOLUTION_SIDE"
            )

        if self.blur_good_laplacian <= self.blur_bad_laplacian:
            raise ValueError(
                "BLUR_GOOD_LAPLACIAN must be greater than BLUR_BAD_LAPLACIAN"
            )

        for name in (
            "low_quality_threshold",
            "valid_threshold",
            "product_match_threshold",
            "strong_mismatch_threshold",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")

        if self.valid_threshold < self.low_quality_threshold:
            raise ValueError(
                "VALID_THRESHOLD must be greater than or equal to LOW_QUALITY_THRESHOLD"
            )

        if self.product_match_threshold < self.strong_mismatch_threshold:
            raise ValueError(
                "PRODUCT_MATCH_THRESHOLD must be greater than or equal to STRONG_MISMATCH_THRESHOLD"
            )

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Scoring weights cannot be negative")

        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"Scoring weights must total 1.0; current total is {total_weight:.4f}"
            )

        return self


settings = Settings()
