"""Capture selected commit-bound evidence files. No third-party code execution."""
import concurrent.futures
import hashlib
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/research/equity_github_reference_20261027_v001'
FILES = {
'NivEyal/TQQQ-Strategy': ['tqqq_tradingview_strategies_compare.py','tqqq_annual_profit_skim_grid_numba.py','dual_momentum_vs_our_strategy.py','ibkr_1p333_etf_results/dual_momentum_vs_our_strategy.csv','tqqq_spy_sma_frozen_audit.py','tqqq_exit_entry_grid_numba.py','tqqq_spy_sma_strategy_robustness.py','requirements.txt','ibkr_1p333_etf_results/tqqq_spy_sma_frozen_audit_summary.csv','ibkr_1p333_etf_results/tqqq_spy_sma_frozen_audit_validation.csv','ibkr_1p333_etf_results/tqqq_spy_sma_frozen_period_audit_summary.csv','ibkr_1p333_etf_results/calmar_spy_sma175_plus4_minus2_to_qqq_transaction_log.csv'],
'swhy0913/Pairs-Trading-with-DTW-Clustering-and-OU-Process': ['new_pairs_trading_DTW.ipynb','Pairs_Identification_DTW_Clustering Algorithm_HKEX100_(Final).ipynb','hkex_stocks_top100pv.csv'],
'Asher-S-Wu/Vectaix-Finance': ['docs/showcase/performance_curve.csv','docs/showcase/model_comparison.csv','docs/showcase/first_cycle_trades.csv','hk_quant/fixed_backtest.py','hk_quant/portfolio.py'],
'ZhongHong98/DCA-VS-Trend-Filtered-Strategy-Backtest-Python': ['Python/DCA_VS_Trend_Filtered_Backtest_Python.py'],
'AI4Finance-Foundation/FinRL-Trading': ['src/backtest/backtest_engine.py'],
'CazSyd/IBS-Strategy': ['scripts/build_readme_charts.py','src/ibs_strategy/data.py','src/ibs_strategy/__init__.py','src/ibs_strategy/backtest.py','src/ibs_strategy/metrics.py']}

def capture(repo):
    p = DATA / repo.replace('/', '__')
    m = json.loads((p/'manifest.json').read_text())
    sha = m['commit']
    for name in FILES[repo]:
        if any(x['path']==name for x in m['files']):
            continue
        url = f'https://raw.githubusercontent.com/{repo}/{sha}/{quote(name)}'
        try:
            with urlopen(Request(url, headers={'User-Agent':'TradingAgents-public-reference-audit'}), timeout=60) as r:
                content = r.read(16*1024*1024+1)
            if len(content)>16*1024*1024:
                raise ValueError('source exceeds 16 MiB')
            dst=p/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(content)
            m['files'].append({'path':name,'url':url,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        except Exception as e:
            m['failures'].append({'path':name,'url':url,'error':str(e)})
    (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    return {'repo':repo,'files':len(m['files']),'failures':m['failures']}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for r in pool.map(capture,FILES): print(json.dumps(r))
