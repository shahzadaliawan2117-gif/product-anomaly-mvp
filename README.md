# Product-Aware Image Anomaly Detection + Quality Evaluation Agent

Runnable FastAPI MVP that independently evaluates **product correctness** and **image quality**. It deliberately avoids an LLM-only architecture: OCR and deterministic CV metrics are the core signals. Components are modular so a production CLIP/SigLIP, detector, segmenter, or OCR engine can replace any service later.

## Architecture
`input → validation → preprocessing/OCR → product metadata matching → deterministic quality → visibility → composition → background → scoring → classification → JSON`

### What is implemented
- Image input: multipart upload, URL, or base64.
- Product metadata: partial JSON accepted; missing fields are omitted from comparisons.
- OCR: Tesseract with preprocessing and confidence.
- Product match: normalized/fuzzy product-name, brand/lab, dosage, variant/form/packaging and conservative category evidence. Optional `reference_image_url` / `reference_image_base64` metadata adds a deterministic HSV + spatial-edge visual descriptor similarity signal, designed to be replaceable by CLIP/SigLIP.
- Quality: Laplacian sharpness, dimensions, brightness histogram, contrast, residual-noise estimate, 8-pixel block-boundary artifact heuristic.
- Visibility: GrabCut foreground localization and frame-edge/area analysis.
- Composition: foreground size, centering, edge margin.
- Background: non-product texture/edge-density heuristic.
- Lighting: brightness clipping and mean-exposure penalty.
- Configurable weighted scoring in `app/config.py`.

> Important limitation: without a client product reference image or a zero-shot vision model, arbitrary SKU identification from appearance alone cannot be guaranteed. This MVP uses OCR + packaging metadata as the principal product-identification evidence. For production, add CLIP/SigLIP zero-shot embeddings and/or a reference-image vector index in `product_matching.py`; the API and classifier do not need to change.

## Installation
Python 3.11+ and Tesseract are required.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install Tesseract separately if it is not already on PATH. Ubuntu/Debian: `sudo apt-get install tesseract-ocr`.

## Run
```bash
uvicorn app.main:app --reload --port 8000
```
Health: `GET http://127.0.0.1:8000/health`
Swagger: `http://127.0.0.1:8000/docs`


## HTML testing dashboard
A standalone browser tester is included as `tester.html`. Start the FastAPI backend first, then double-click `tester.html` and keep the API Base URL as `http://127.0.0.1:8000`. It supports file upload, image URL, base64 input, editable product JSON, health checks, scores, detected fields, issues, explanations, and the raw JSON response.

### Windows Tesseract setup
On Windows 11, a convenient installation command is:
```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```
If `tesseract --version` is not recognized after installation, add `C:\Program Files\Tesseract-OCR` to the user PATH and restart VS Code/PowerShell.

## API
### Multipart `POST /analyze`
Exactly one image source is accepted (`image`, `image_url`, or `image_base64`) plus `product_json`.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F 'image=@product.jpg' \
  -F 'product_json={"id":"12345","name":"Doliprane 1000 mg","category":"Medicament","brand":"Sanofi","dosage":"1000 mg"}'
```

### JSON convenience `POST /analyze-json`
```json
{
  "image": {"url": "https://example.com/product.jpg"},
  "product_json": {"id":"12345","name":"Doliprane 1000 mg","brand":"Sanofi"}
}
```
Use `{"image":{"base64":"..."}, ...}` for base64.

## Scores
All scores are 0–100.
- `sharpness`: linear normalization of Laplacian variance between configurable bad/good thresholds.
- `resolution`: `100 × min(width,height) / target_side`, clipped to 100.
- `lighting`: starts at 100 and subtracts penalties for clipped dark/bright pixels and mean brightness far from mid-range.
- `image_quality`: `0.27 sharpness + 0.18 resolution + 0.20 lighting + 0.13 contrast + 0.12 noise + 0.10 compression`.
- `visibility`: foreground-area and cutoff penalties.
- `composition`: foreground-size, centering, and edge-margin penalties.
- `background`: background standard deviation and edge-density penalty.
- `text_readability`: scaled OCR confidence.
- `product_match`: weighted normalized evidence from available metadata only; missing attributes are not scored as failures.
- `overall`: configurable weighted mean from `Settings.weights`. Product match is intentionally **not** mixed into image-quality overall; correctness is classified independently.

## Classification logic
Specific confirmed metadata mismatches take precedence: `WRONG_CATEGORY`, `WRONG_BRAND`, `WRONG_VARIANT`, then `WRONG_PRODUCT`. If product evidence is consistent, visual metrics produce `VALID`, `VALID_BUT_LOW_QUALITY`, or `ANOMALY`. Thus a blurry correct product does not automatically become a wrong product, while a high-quality wrong product cannot become valid.

## Tests
```bash
pytest -q
```
Tests cover health, correct product, wrong product/brand/variant/category, blur, low resolution, composition, background, and unreadable text using generated legal synthetic fixtures.

## Error handling
Clean HTTP errors are returned for empty/corrupt/unsupported images, missing or invalid metadata JSON, inaccessible URLs, invalid base64, and ambiguous multiple image inputs. OCR failure degrades the OCR signal instead of crashing the service.

## Docker
```bash
docker build -t product-anomaly-mvp .
docker run --rm -p 8000:8000 product-anomaly-mvp
```

## Known limitations
- SKU-level visual recognition from a single image + textual metadata is fundamentally open-set. The MVP supports optional reference-image visual similarity, but production matching should use a labeled reference-image vector index plus a pretrained zero-shot image-text model such as SigLIP/CLIP.
- GrabCut can fail on complex scenes or products touching borders.
- Tesseract accuracy depends on language, print size, glare, orientation, and typography.
- Thresholds require calibration on real client data; no 100% accuracy claim is made.
- Background/compression/noise detectors are practical heuristics, not learned perceptual-quality models.

## Production scaling recommendations
Add SigLIP/CLIP embedding service, reference-image vector DB, YOLO/RT-DETR packaging detector, segmentation model, PaddleOCR multilingual OCR, async model workers, GPU inference, per-category calibrated thresholds, observability, authentication/rate limits, artifact storage, and a labeled evaluation set with precision/recall metrics.


## One-link client deployment (Render)

The project is deployment-ready as a single web service. The FastAPI server serves the testing dashboard at `/`, the API at `/analyze`, Swagger at `/docs`, and the health check at `/health`. The dashboard automatically uses the same public origin when hosted, so the client only needs one URL.

For Render, deploy the repository as a **Web Service** using **Docker**. The Dockerfile installs the Tesseract system package and starts Uvicorn on Render's `PORT` environment variable. Set the health check path to `/health`.
