import base64
import io
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.config import settings


ALLOWED = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
MAX_REDIRECTS = 4
MAX_PIXELS = 40_000_000
CHUNK_SIZE = 64 * 1024

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 ProductImageQA/1.0"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


def _is_blocked_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_public_url(url: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid image URL: {exc}")

    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(400, "Image URL must use http or https")

    if parsed.username or parsed.password:
        raise HTTPException(400, "Image URL must not contain embedded credentials")

    if not parsed.hostname:
        raise HTTPException(400, "Image URL must contain a hostname")

    try:
        port = parsed.port
    except ValueError:
        raise HTTPException(400, "Invalid image URL port")

    if port is not None and port not in {80, 443}:
        raise HTTPException(400, "Only standard HTTP/HTTPS ports are allowed")

    hostname = parsed.hostname.rstrip(".").lower()

    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise HTTPException(400, "Private or local image URLs are not allowed")

    try:
        addr_info = socket.getaddrinfo(
            hostname,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        raise HTTPException(400, "Image URL hostname could not be resolved")

    resolved_ips = {item[4][0] for item in addr_info}
    if not resolved_ips:
        raise HTTPException(400, "Image URL hostname could not be resolved")

    if any(_is_blocked_ip(ip) for ip in resolved_ips):
        raise HTTPException(400, "Private, local, or reserved image URLs are not allowed")


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

            width, height = im.size
            if width <= 0 or height <= 0:
                raise HTTPException(400, "Invalid image dimensions")

            if width * height > MAX_PIXELS:
                raise HTTPException(413, "Image dimensions are too large")

            rgb = im.convert("RGB")
            arr = np.asarray(rgb)

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid or corrupted image: {exc}")

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _download_url(url: str) -> bytes:
    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):
        _validate_public_url(current_url)

        try:
            response = requests.get(
                current_url,
                timeout=settings.request_timeout_seconds,
                stream=True,
                allow_redirects=False,
                headers=REQUEST_HEADERS,
            )
        except requests.Timeout:
            raise HTTPException(408, "Image URL request timed out")
        except requests.RequestException as exc:
            raise HTTPException(400, f"Image URL unavailable: {exc}")

        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()

            if not location:
                raise HTTPException(400, "Image URL returned an invalid redirect")

            if redirect_count >= MAX_REDIRECTS:
                raise HTTPException(400, "Image URL exceeded the redirect limit")

            current_url = urljoin(current_url, location)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = response.status_code
            response.close()
            raise HTTPException(
                400,
                f"Image URL returned HTTP {status}. The remote host may block server-side downloads.",
            ) from exc

        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type.startswith("text/") or content_type in {
            "application/json",
            "application/xml",
            "text/html",
        }:
            response.close()
            raise HTTPException(415, f"URL did not return an image ({content_type or 'unknown type'})")

        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > settings.max_image_bytes:
                    response.close()
                    raise HTTPException(413, "Remote image exceeds maximum allowed size")
            except ValueError:
                pass

        data = bytearray()

        try:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > settings.max_image_bytes:
                    raise HTTPException(413, "Remote image exceeds maximum allowed size")
        except HTTPException:
            raise
        except requests.RequestException as exc:
            raise HTTPException(400, f"Image download failed: {exc}")
        finally:
            response.close()

        return bytes(data)

    raise HTTPException(400, "Image URL exceeded the redirect limit")


def from_url(url: str) -> np.ndarray:
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(400, "Image URL is required")

    data = _download_url(url.strip())
    return _decode_bytes(data)


def from_base64(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(400, "Base64 image is required")

    value = value.strip()

    if value.startswith("data:"):
        if "," not in value:
            raise HTTPException(400, "Invalid data URL")
        header, value = value.split(",", 1)
        if ";base64" not in header.lower():
            raise HTTPException(400, "Data URL must contain base64 image data")

    value = re.sub(r"\s+", "", value)

    max_encoded_length = ((settings.max_image_bytes + 2) // 3) * 4 + 8
    if len(value) > max_encoded_length:
        raise HTTPException(413, "Base64 image exceeds maximum allowed size")

    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid base64 image: {exc}")

    return _decode_bytes(data)


def from_bytes(data: bytes) -> np.ndarray:
    return _decode_bytes(data)
