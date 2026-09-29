import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io, cv2, numpy as np, pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client(): return TestClient(app)

def make_product(text='DOLIPRANE 1000 mg SANOFI', size=(1200,900), blur=0, small=False, clutter=False):
    img=Image.new('RGB',size,'white'); d=ImageDraw.Draw(img)
    box=(250,180,950,720) if not small else (500,350,700,500)
    if clutter:
        for i in range(25): d.line((0,i*35,size[0],(i*35+300)%size[1]),fill=(120,120,120),width=3)
    d.rectangle(box, fill=(235,235,235), outline='black', width=8)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',52 if not small else 20)
    except: font=None
    d.text((box[0]+25,box[1]+80),text,fill='black',font=font)
    arr=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    if blur: arr=cv2.GaussianBlur(arr,(blur,blur),0)
    ok,enc=cv2.imencode('.png',arr); return enc.tobytes()

def post(client,img,meta):
    import json
    return client.post('/analyze',files={'image':('x.png',img,'image/png')},data={'product_json':json.dumps(meta)})
