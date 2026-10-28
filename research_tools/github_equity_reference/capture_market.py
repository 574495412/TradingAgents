"""Freeze public Yahoo chart bytes before source-engine replay (no credentials)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/research/equity_github_reference_20261027_v001'
OUT=DATA/'market'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    params={'period1':int(datetime(2010,2,9,tzinfo=timezone.utc).timestamp()),'period2':int(datetime(2026,8,1,tzinfo=timezone.utc).timestamp()),'interval':'1d','events':'div,splits'}
    url='https://query1.finance.yahoo.com/v8/finance/chart/TQQQ?'+urlencode(params)
    record={'symbol':'TQQQ','url':url,'captured_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as r: data=r.read()
        payload=json.loads(data)
        result=payload['chart']['result'][0]
        record.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),timestamps=len(result['timestamp']))
        (OUT/'TQQQ_yahoo_chart.json').write_bytes(data)
    except Exception as exc:
        record['error']=str(exc)
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
