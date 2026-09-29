import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app.config import settings
from app.models.schemas import AnalyzeResponse, ProductMetadata
from app.services.pipeline import analyze
from app.utils.image_loader import from_base64, from_bytes, from_url


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("product_anomaly_api")

app = FastAPI(
    title="Product-Aware Image Anomaly Detection + Quality Evaluation Agent",
    version="1.0.0",
)

TESTER_PATH = Path(__file__).resolve().parents[1] / "tester.html"
UPLOAD_CHUNK_SIZE = 64 * 1024


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "Unhandled request error request_id=%s method=%s path=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "Request validation failed method=%s path=%s errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID")
    logger.exception(
        "Unexpected server error request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/", include_in_schema=False)
def tester():
    if not TESTER_PATH.exists():
        raise HTTPException(404, "Tester UI is unavailable")
    return FileResponse(TESTER_PATH)


@app.get("/health")
def health():
    # Preserve the original public API contract required by the client.
    return {"status": "ok"}


def parse_meta(raw: Any) -> ProductMetadata:
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)

        if not isinstance(raw, dict):
            raise TypeError("product_json must be a JSON object")

        return ProductMetadata.model_validate(raw)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            f"Invalid product_json: malformed JSON at line {exc.lineno}, column {exc.colno}",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail={
                "message": "Invalid product_json",
                "errors": exc.errors(),
            },
        ) from exc
    except TypeError as exc:
        raise HTTPException(422, f"Invalid product_json: {exc}") from exc


async def _read_upload_limited(image: UploadFile) -> bytes:
    data = bytearray()

    while True:
        chunk = await image.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break

        data.extend(chunk)

        if len(data) > settings.max_image_bytes:
            raise HTTPException(413, "Image exceeds maximum allowed size")

    if not data:
        raise HTTPException(400, "Empty image upload")

    return bytes(data)


def _run_analysis(arr, meta: ProductMetadata) -> AnalyzeResponse:
    try:
        return analyze(arr, meta)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Analysis pipeline failed product_id=%s",
            getattr(meta, "id", None),
        )
        raise HTTPException(500, "Analysis pipeline failed") from exc


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_multipart(
    image: Optional[UploadFile] = File(None),
    product_json: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
):
    if not product_json:
        raise HTTPException(422, "Missing product_json")

    provided = sum(
        bool(source)
        for source in [
            image is not None,
            image_url and image_url.strip(),
            image_base64 and image_base64.strip(),
        ]
    )

    if provided != 1:
        raise HTTPException(
            422,
            "Provide exactly one image source: file, image_url, or image_base64",
        )

    meta = parse_meta(product_json)

    if image is not None:
        data = await _read_upload_limited(image)
        arr = from_bytes(data)
    elif image_url:
        arr = from_url(image_url)
    else:
        arr = from_base64(image_base64 or "")

    return _run_analysis(arr, meta)


@app.post("/analyze-json", response_model=AnalyzeResponse)
def analyze_json(payload: dict = Body(...)):
    if "product_json" not in payload:
        raise HTTPException(422, "Missing product_json")

    meta = parse_meta(payload["product_json"])

    image_input = payload.get("image")
    if not isinstance(image_input, dict):
        raise HTTPException(
            422,
            "'image' must contain exactly one of url or base64",
        )

    keys = [
        key
        for key in ("url", "base64")
        if isinstance(image_input.get(key), str) and image_input.get(key).strip()
    ]

    if len(keys) != 1:
        raise HTTPException(
            422,
            "'image' must contain exactly one of url or base64",
        )

    if keys[0] == "url":
        arr = from_url(image_input["url"])
    else:
        arr = from_base64(image_input["base64"])

    return _run_analysis(arr, meta)
