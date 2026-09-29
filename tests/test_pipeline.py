from conftest import make_product, post
BASE={'id':'12345','name':'Doliprane 1000 mg','category':'Medicament','brand':'Sanofi','dosage':'1000 mg','form':'Comprime'}

def test_health(client): assert client.get('/health').json()=={'status':'ok'}
def test_correct_product(client):
    r=post(client,make_product(),BASE); assert r.status_code==200; j=r.json(); assert j['status'] in {'VALID','ANOMALY'}; assert j['is_correct_product'] is True

def test_wrong_product(client):
    r=post(client,make_product('MAALOX ANTACID 400 mg PHARMA'),BASE); j=r.json(); assert j['status'] in {'WRONG_PRODUCT','WRONG_BRAND','WRONG_VARIANT'}; assert not j['is_correct_product']

def test_wrong_brand(client):
    meta={**BASE,'brand':'Sanofi'}; r=post(client,make_product('DOLIPRANE 1000 mg ACME'),meta); j=r.json(); assert j['status'] in {'WRONG_BRAND','WRONG_PRODUCT'}

def test_wrong_variant(client):
    r=post(client,make_product('DOLIPRANE 500 mg SANOFI'),BASE); assert r.json()['status']=='WRONG_VARIANT'

def test_wrong_category(client):
    meta={**BASE,'category':'Shampoo','name':'Fresh Hair Shampoo','brand':'Fresh'}
    r=post(client,make_product('DOLIPRANE 1000 mg TABLET'),meta); assert r.json()['is_correct_product'] is False

def test_blurry_correct_not_wrong_product(client):
    r=post(client,make_product(blur=41),BASE); j=r.json(); assert not j['status'].startswith('WRONG_')

def test_low_resolution(client):
    r=post(client,make_product(size=(240,180)),BASE); j=r.json(); assert j['scores']['resolution']<50

def test_poor_composition(client):
    r=post(client,make_product(small=True),BASE); j=r.json(); assert j['scores']['composition']<80

def test_bad_background(client):
    r=post(client,make_product(clutter=True),BASE); j=r.json(); assert 'background' in j['scores']

def test_unreadable_text(client):
    meta={'id':'x','name':'Doliprane 1000 mg'}; r=post(client,make_product('',blur=51),meta); j=r.json(); assert 'text_readability' in j['scores']
