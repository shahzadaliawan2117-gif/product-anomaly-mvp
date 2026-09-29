import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import ValidationError
from app.models.schemas import ProductMetadata, AnalyzeResponse
from app.utils.image_loader import from_bytes, from_url, from_base64
from app.services.pipeline import analyze

app=FastAPI(title='Product-Aware Image Anomaly Detection + Quality Evaluation Agent',version='0.1.0')

@app.get('/', include_in_schema=False)
def tester():
    return FileResponse(Path(__file__).resolve().parents[1] / 'tester.html')

@app.get('/health')
def health(): return {'status':'ok'}

def parse_meta(raw)->ProductMetadata:
    try:
        if isinstance(raw,str): raw=json.loads(raw)
        return ProductMetadata.model_validate(raw)
    except (json.JSONDecodeError,ValidationError,TypeError) as e:
        raise HTTPException(422,f'Invalid product_json: {e}')

@app.post('/analyze',response_model=AnalyzeResponse)
async def analyze_multipart(image: Optional[UploadFile]=File(None), product_json: Optional[str]=Form(None), image_url: Optional[str]=Form(None), image_base64: Optional[str]=Form(None)):
    if not product_json: raise HTTPException(422,'Missing product_json')
    provided=sum(x is not None for x in [image,image_url,image_base64])
    if provided!=1: raise HTTPException(422,'Provide exactly one image source: file, image_url, or image_base64')
    meta=parse_meta(product_json)
    if image is not None: arr=from_bytes(await image.read())
    elif image_url is not None: arr=from_url(image_url)
    else: arr=from_base64(image_base64)
    return analyze(arr,meta)

@app.post('/analyze-json',response_model=AnalyzeResponse,include_in_schema=True)
def analyze_json(payload: dict=Body(...)):
    # Convenience JSON transport for URL/base64 while preserving the same two conceptual inputs.
    if 'product_json' not in payload: raise HTTPException(422,'Missing product_json')
    meta=parse_meta(payload['product_json'])
    image_input=payload.get('image')
    if not isinstance(image_input,dict): raise HTTPException(422,"'image' must contain exactly one of url or base64")
    keys=[k for k in ['url','base64'] if image_input.get(k)]
    if len(keys)!=1: raise HTTPException(422,"'image' must contain exactly one of url or base64")
    arr=from_url(image_input['url']) if keys[0]=='url' else from_base64(image_input['base64'])
    return analyze(arr,meta)
