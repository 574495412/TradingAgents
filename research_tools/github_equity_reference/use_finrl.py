"""Use FinRL's original AdaptiveRotation engine/config; no strategy rewrite.

Local research only: capture free data or emit original strategy target weights.
No application .env, broker module, API credential or order executor is loaded.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/research/equity_mature_adoption_20261027_v001'
UPSTREAM=DATA/'FinRL-Trading'
CONFIG=UPSTREAM/'src/strategies/AdaptiveRotationConf_v1.2.2.yaml'
sys.path.insert(0,str(DATA/'runtime_deps'))
sys.path.insert(0,str(UPSTREAM))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config():
    from src.strategies.adaptive_rotation.config_loader import load_config as original
    return original(str(CONFIG))


def capture():
    import pandas as pd
    config=load_config()
    folder=DATA/'market'
    folder.mkdir(exist_ok=True)
    previous=folder/'manifest.json'
    if previous.exists() and not (folder/'initial-capture.json').exists():
        (folder/'initial-capture.json').write_bytes(previous.read_bytes())
    symbols=sorted(set(config.get_required_symbols()+config.portfolio.fallback.symbols))
    params={'period1':int(datetime(2017,1,1,tzinfo=timezone.utc).timestamp()),
            'period2':int(datetime(2026,10,24,tzinfo=timezone.utc).timestamp()),'interval':'1d','events':'div,splits'}
    def one(symbol):
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?{urlencode(params)}'
        receipt={'symbol':symbol,'url':url,'captured_at_utc':datetime.now(timezone.utc).isoformat()}
        try:
            raw_path=folder/f'{symbol}_chart.json'
            raw=raw_path.read_bytes() if raw_path.exists() else urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read()
            raw_path.write_bytes(raw)
            obj=json.loads(raw)['chart']['result'][0]
            idx=pd.to_datetime(obj['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
            q=pd.DataFrame(obj['indicators']['quote'][0],index=idx)
            adj=pd.Series(obj['indicators']['adjclose'][0]['adjclose'],index=idx)
            missing=q[['open','high','low','close']].isna().any(axis=1) | adj.isna()
            if missing.any():
                import pandas_market_calendars as mcal
                sessions=mcal.get_calendar('NYSE').schedule(start_date=idx[0],end_date=idx[-1]).index
                if symbol!='^VIX' or idx[missing].isin(sessions).any():raise ValueError('missing quote on exchange session')
                receipt['excluded_no_quote_nonexchange_sessions']=[str(d.date()) for d in idx[missing]]
                q=q.loc[~missing];adj=adj.loc[~missing];idx=q.index
            if not idx.is_unique or not idx.is_monotonic_increasing:raise ValueError('invalid date index')
            factor=adj/q.close
            frame=q[['open','high','low','close']].mul(factor,axis=0)
            frame['volume']=q.volume
            frame.index.name='date'
            dst=folder/f'{symbol}_daily.csv';frame.to_csv(dst,float_format='%.15g')
            receipt.update(raw_sha256=hashlib.sha256(raw).hexdigest(),csv_sha256=sha(dst),rows=len(frame),start=str(idx[0].date()),end=str(idx[-1].date()))
        except Exception as e:receipt['error']=str(e)
        return receipt
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:receipts=list(pool.map(one,symbols))
    result={'data_source':'Yahoo adjusted OHLC, volume unchanged; replaces upstream FMP input without altering strategy',
            'same_as_author_historical_snapshot':False,'config_sha256':sha(CONFIG),'symbols':receipts}
    save(folder/'manifest.json',result)
    failures=[r for r in receipts if 'error' in r]
    print(json.dumps({'captured':len(receipts)-len(failures),'failed':failures}))
    if failures:raise RuntimeError('capture incomplete; failures preserved')


def run(date,output):
    from dataclasses import asdict
    from src.strategies.adaptive_rotation import AdaptiveRotationEngine
    from src.strategies.adaptive_rotation.data_preprocessor import DataPreprocessor
    from importlib.metadata import version
    output.mkdir(parents=True,exist_ok=False)
    config=load_config()
    manifest=json.loads((DATA/'market/manifest.json').read_text())
    for item in manifest['symbols']:
        if 'error' in item:raise ValueError(f'incomplete market capture: {item["symbol"]}')
        if sha(DATA/'market'/f'{item["symbol"]}_daily.csv')!=item['csv_sha256']:raise ValueError('market input changed')
    source_files={str(p.relative_to(UPSTREAM)):sha(p) for p in (UPSTREAM/'src/strategies/adaptive_rotation').rglob('*.py')}
    save(output/'registration.json',{'source':json.loads((DATA/'source.json').read_text()),
          'source_files':source_files,'config_sha256':sha(CONFIG),'config_unmodified':True,
          'market_manifest_sha256':sha(DATA/'market/manifest.json'),'run_date':date,
          'entrypoint_sha256':sha(Path(__file__)),'runtime':{'pandas':version('pandas'),'numpy':version('numpy'),'pydantic':version('pydantic'),'pandas_market_calendars':version('pandas_market_calendars')},
          'orders_enabled':False,'exact_author_headline_match_required':False})
    preprocessor=DataPreprocessor(config)
    preprocessor.load_and_prepare(data_dir=str(DATA/'market'))
    engine=AdaptiveRotationEngine(config=config,data_preprocessor=preprocessor)
    prices={symbol:frame['close'] for symbol,frame in preprocessor.get_data_as_of(date).items()}
    weights,audit=engine.run(price_data=prices,as_of_date=date,mode='backtest')
    audit.to_json(str(output/'native_audit.json'))
    save(output/'native_weights.json',asdict(weights))
    result={'schema':'equity-native-adoption-result.v1','repo':'AI4Finance-Foundation/FinRL-Trading',
            'commit':json.loads((DATA/'source.json').read_text())['commit'],'strategy':'AdaptiveRotation v1.2.2',
            'native_engine_executed':True,'strategy_code_changed':False,'strategy_configuration_changed':False,
            'signal_date':date,'data_symbols_loaded':len(prices),'target_weights':weights.weights,
            'cash_weight':weights.cash_weight,'regime_state':weights.regime_state,
            'headline_return_reproduced':False,'historical_profit_evaluated':False,
            'orders_enabled':False,'live_deployed':False,
            'method':'direct_original_engine_call_no_custom_strategy_or_position_cap',
            'source_data_deviation':manifest['data_source']}
    save(output/'result.json',result)
    save(ROOT/'docs/research/equities_github_reference_20261027/adoption-verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--date',default='2026-10-23')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.capture:capture()
    else:
        if args.output is None:parser.error('--output is required for native engine run')
        run(args.date,args.output.resolve())
