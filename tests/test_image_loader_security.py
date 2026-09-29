import base64
import io
import socket

import pytest
from fastapi import HTTPException
from PIL import Image

from app.config import settings
from app.utils import image_loader


class FakeResponse:
    def __init__(self, data=b"", status_code=200, headers=None):
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


def make_png_bytes(width=32, height=32):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_rejects_non_http_url():
    with pytest.raises(HTTPException) as exc:
        image_loader.from_url("file:///etc/passwd")
    assert exc.value.status_code == 400


def test_rejects_localhost_url():
    with pytest.raises(HTTPException) as exc:
        image_loader.from_url("http://localhost/image.png")
    assert exc.value.status_code == 400
    assert "local" in str(exc.value.detail).lower() or "private" in str(exc.value.detail).lower()


def test_rejects_private_ip_url():
    with pytest.raises(HTTPException) as exc:
        image_loader.from_url("http://127.0.0.1/image.png")
    assert exc.value.status_code == 400


def test_rejects_invalid_base64():
    with pytest.raises(HTTPException) as exc:
        image_loader.from_base64("not-valid-base64%%%")
    assert exc.value.status_code == 400


def test_rejects_oversized_base64():
    raw = b"x" * (settings.max_image_bytes + 1)
    encoded = base64.b64encode(raw).decode("ascii")

    with pytest.raises(HTTPException) as exc:
        image_loader.from_base64(encoded)

    assert exc.value.status_code == 413


def test_public_url_download_succeeds(monkeypatch):
    image_bytes = make_png_bytes()

    def fake_getaddrinfo(host, port, type=0, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ]

    def fake_get(*args, **kwargs):
        return FakeResponse(
            data=image_bytes,
            status_code=200,
            headers={
                "Content-Type": "image/png",
                "Content-Length": str(len(image_bytes)),
            },
        )

    monkeypatch.setattr(image_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(image_loader.requests, "get", fake_get)

    arr = image_loader.from_url("https://example.com/product.png")

    assert arr is not None
    assert arr.shape == (32, 32, 3)


def test_html_response_is_rejected(monkeypatch):
    def fake_getaddrinfo(host, port, type=0, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ]

    def fake_get(*args, **kwargs):
        return FakeResponse(
            data=b"<html>not an image</html>",
            status_code=200,
            headers={"Content-Type": "text/html"},
        )

    monkeypatch.setattr(image_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(image_loader.requests, "get", fake_get)

    with pytest.raises(HTTPException) as exc:
        image_loader.from_url("https://example.com/not-image")

    assert exc.value.status_code == 415
