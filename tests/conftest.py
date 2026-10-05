import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _test_font(size: int):
    """Return a readable bold font on Windows/Linux/macOS when available.

    The old fixture used only a Linux DejaVu path. On Windows that path fails,
    Pillow falls back to its tiny default font, and OCR cannot reliably read the
    synthetic product text. That made wrong-brand pipeline tests depend on the
    developer OS rather than on application logic.
    """
    candidates = [
        # Windows
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
        # Linux
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        # macOS
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]

    for path in candidates:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size)
        except OSError:
            pass

    # Last-resort Pillow font. Tests remain runnable, although OCR quality can
    # be lower on machines without any of the common system fonts above.
    return ImageFont.load_default()


def make_product(
    text="DOLIPRANE 1000 mg SANOFI",
    size=(1200, 900),
    blur=0,
    small=False,
    clutter=False,
):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)

    box = (250, 180, 950, 720) if not small else (500, 350, 700, 500)

    if clutter:
        for i in range(25):
            d.line(
                (0, i * 35, size[0], (i * 35 + 300) % size[1]),
                fill=(120, 120, 120),
                width=3,
            )

    d.rectangle(box, fill=(235, 235, 235), outline="black", width=8)

    font = _test_font(52 if not small else 20)
    d.text(
        (box[0] + 25, box[1] + 80),
        text,
        fill="black",
        font=font,
    )

    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    if blur:
        arr = cv2.GaussianBlur(arr, (blur, blur), 0)

    ok, enc = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Could not encode synthetic test image")

    return enc.tobytes()


def post(client, img, meta):
    import json

    return client.post(
        "/analyze",
        files={"image": ("x.png", img, "image/png")},
        data={"product_json": json.dumps(meta)},
    )
