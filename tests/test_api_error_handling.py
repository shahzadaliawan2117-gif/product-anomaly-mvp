import io
import json

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module


client = TestClient(main_module.app)


def make_png_bytes(width=32, height=32):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


VALID_PRODUCT = {
    "id": "12345",
    "name": "Doliprane 1000 mg Comprime",
    "category": "Medicament",
    "brand": "Sanofi",
    "dosage": "1000 mg",
    "form": "Comprime",
}


def test_health_preserves_contract():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_product_json_returns_422():
    response = client.post(
        "/analyze",
        files={"image": ("product.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Missing product_json"


def test_malformed_product_json_returns_422():
    response = client.post(
        "/analyze",
        data={"product_json": "{bad-json"},
        files={"image": ("product.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert "malformed JSON" in str(response.json()["detail"])


def test_requires_exactly_one_image_source():
    response = client.post(
        "/analyze",
        data={
            "product_json": json.dumps(VALID_PRODUCT),
            "image_url": "https://example.com/product.png",
        },
        files={"image": ("product.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert "exactly one image source" in response.json()["detail"]


def test_analyze_json_rejects_missing_image():
    response = client.post(
        "/analyze-json",
        json={"product_json": VALID_PRODUCT},
    )
    assert response.status_code == 422
    assert "exactly one of url or base64" in response.json()["detail"]


def test_unexpected_pipeline_error_returns_safe_500(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sensitive internal failure")

    monkeypatch.setattr(main_module, "analyze", boom)

    response = client.post(
        "/analyze",
        data={"product_json": json.dumps(VALID_PRODUCT)},
        files={"image": ("product.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Analysis pipeline failed"
    assert "sensitive internal failure" not in response.text
