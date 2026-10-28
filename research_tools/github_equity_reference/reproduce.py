"""Source-original return checks and one fixed IBS rule reproduction.

Run after capture_sources.py, capture_details.py and capture_market.py.
Downloaded source remains local data; only reviewed metrics/backtest modules are
loaded, bypassing package initialization, downloads, optimizers and broker code.
No application configuration, credentials or trading connections are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/research/equity_github_reference_20261027_v001'
DOC = ROOT / 'docs/research/equities_github_reference_20261027'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    receipts = []
    for directory in sorted(DATA.iterdir()):
        if not directory.is_dir() or directory.name == 'market':
            continue
        manifest = json.loads((directory / 'manifest.json').read_text())
        for item in manifest['files']:
            if digest(directory / item['path']) != item['sha256']:
                raise ValueError(f'source digest mismatch: {directory.name}/{item["path"]}')
        receipts.append(manifest)
    market = json.loads((DATA / 'market/manifest.json').read_text())
    if digest(DATA / 'market/TQQQ_yahoo_chart.json') != market['sha256']:
        raise ValueError('market bytes changed')
    return receipts, market


def load_market():
    obj = json.loads((DATA / 'market/TQQQ_yahoo_chart.json').read_text())['chart']['result'][0]
    index = pd.to_datetime(obj['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    q = pd.DataFrame(obj['indicators']['quote'][0], index=index)
    adj = pd.Series(obj['indicators']['adjclose'][0]['adjclose'], index=index)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError('unordered or duplicate bars')
    if q[['open', 'high', 'low', 'close']].isna().any().any() or adj.isna().any():
        raise ValueError('missing prices; do not silently drop or fill bars')
    factor = adj / q['close']
    bars = q[['open', 'high', 'low', 'close']].mul(factor, axis=0)
    bars.columns = ['Open', 'High', 'Low', 'Close']
    bars.index.name = 'Date'
    if not ((bars.High >= bars[['Open', 'Close', 'Low']].max(axis=1)) &
            (bars.Low <= bars[['Open', 'Close', 'High']].min(axis=1)) & (bars.Low > 0)).all():
        raise ValueError('invalid OHLC')
    bars['IBS'] = (bars.Close - bars.Low) / (bars.High - bars.Low).replace(0, np.nan)
    return bars


def load_original_engine():
    folder = DATA / 'CazSyd__IBS-Strategy/src/ibs_strategy'
    # A minimal namespace avoids executing upstream __init__.py (which imports
    # downloads, visualization, optimizers and live helpers). These two source
    # files have been reviewed and have only numerical/stdlib imports.
    package = types.ModuleType('_frozen_ibs_reference')
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    for filename in ['metrics', 'backtest']:
        name = package.__name__ + '.' + filename
        spec = importlib.util.spec_from_file_location(name, folder / f'{filename}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[package.__name__ + '.backtest']


def replay_independently(bars, entry, exit_threshold, initial_cash):
    """Independent, narrow implementation of the author's fixed full/open rule."""
    cash, held = float(initial_cash), 0
    rows = []
    trades = []
    opened = None
    for i, (date, bar) in enumerate(bars.iterrows()):
        if i:
            signal = bars.IBS.iloc[i - 1]
            if held:
                if signal > exit_threshold:
                    cash += held * bar.Open
                    trades.append({**opened, 'exit_date': date.isoformat(), 'exit_price': float(bar.Open)})
                    held = 0
                    opened = None
            elif signal < entry:
                quantity = int(cash // bar.Open)
                if quantity > 0:
                    cash -= quantity * bar.Open
                    held = quantity
                    opened = {'entry_date': date.isoformat(), 'entry_price': float(bar.Open), 'shares': quantity}
        rows.append({'date': date, 'cash': cash, 'shares': held, 'equity': cash + held * bar.Close})
    if opened:
        trades.append({**opened, 'exit_date': None, 'exit_price': None})
    return pd.DataFrame(rows).set_index('date'), trades


def metrics(equity):
    returns = equity.pct_change().fillna(0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return {'start': equity.index[0].date().isoformat(), 'end': equity.index[-1].date().isoformat(),
            'rows': len(equity), 'final_equity': float(equity.iloc[-1]),
            'total_return': float(equity.iloc[-1] / equity.iloc[0] - 1),
            'cagr': float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
            'max_drawdown': float((equity / equity.cummax() - 1).min()),
            'sharpe_252_zero_rf_ddof1': float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))}


def author_hk_curve_check(tolerances):
    folder = DATA / 'Asher-S-Wu__Vectaix-Finance/docs/showcase'
    curve = pd.read_csv(folder / 'performance_curve.csv', parse_dates=['date'])
    claims = pd.read_csv(folder / 'model_comparison.csv')
    days = (curve.date.iloc[-1] - curve.date.iloc[0]).days
    checks = []
    for model in ['factor', 'linear', 'lightgbm_small', 'lightgbm_large']:
        row = claims[(claims.model == model) & (claims.scenario == 'normal')].iloc[0]
        values = pd.concat([pd.Series([1.0]), curve[model]], ignore_index=True)
        reconstructed = {'total_return': float(values.iloc[-1] - 1),
                         'cagr': float(values.iloc[-1] ** (365.25 / days) - 1),
                         'max_drawdown_daily': float((values / values.cummax() - 1).min())}
        errors = {k: abs(reconstructed[k] - float(row[k])) for k in reconstructed}
        passed = (errors['total_return'] <= tolerances['HK_artifact_total_return_abs'] and
                  errors['cagr'] <= tolerances['HK_artifact_CAGR_abs'] and
                  errors['max_drawdown_daily'] <= tolerances['HK_artifact_drawdown_abs'])
        checks.append({'model': model, 'rows': len(curve), 'start': str(curve.date.iloc[0].date()),
                       'end': str(curve.date.iloc[-1].date()), 'errors': errors, 'reconstructed': reconstructed,
                       'published': {k: float(row[k]) for k in reconstructed},
                       'arithmetic_passed': passed, 'training_reproduced': False,
                       'full_trade_replay_reproduced': False, 'execution_validated': bool(row.execution_validated),
                       'reference_valuation_days': int(row.reference_valuation_days),
                       'open_positions': int(row.open_positions), 'pending_sells': int(row.pending_sells),
                       'realized_account_return_certified': False})
    return checks


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((DOC / 'plan.json').read_text())
    receipts, market = verify_sources()
    save(output / 'registration.json', {'plan': plan, 'plan_sha256': digest(DOC / 'plan.json'),
                                        'source_manifest_sha256': {x['repo']: digest(DATA / x['repo'].replace('/', '__') / 'manifest.json') for x in receipts},
                                        'market': market, 'reproduction_script_sha256': digest(Path(__file__))})
    bars = load_market()
    bars.to_csv(output / 'adjusted_bars.csv', float_format='%.15g')
    config = plan['IBS_replication']
    original = load_original_engine().run_backtest(bars, entry_threshold=config['entry_threshold'],
                                                  exit_threshold=config['exit_threshold'],
                                                  initial_capital=config['initial_cash'], cash_rate=0,
                                                  entry_fill='open', exit_fill='open', position_sizing='full')
    replica, trades = replay_independently(bars, config['entry_threshold'], config['exit_threshold'], config['initial_cash'])
    ref = original.data[['Cash', 'Shares', 'Capital']].rename(columns={'Cash': 'cash', 'Shares': 'shares', 'Capital': 'equity'})
    errors = (ref - replica).abs().max().to_dict()
    original_trades = []
    for trade in original.trades:
        row = asdict(trade)
        row['entry_date'] = row['entry_date'].isoformat()
        row['exit_date'] = row['exit_date'].isoformat() if row['exit_date'] is not None else None
        original_trades.append(row)
    equal_trades = original_trades == trades
    tol = plan['tolerances_registered_before_execution']
    parity = (errors['equity'] <= tol['same_input_daily_equity_abs_USD'] and
              errors['cash'] <= tol['same_input_cash_abs_USD'] and errors['shares'] == 0 and equal_trades)
    # Causality control replays the same frozen strategy on a prefix. This is
    # an engineering check, not another economic candidate or out-of-sample test.
    prefix = bars.iloc[:len(bars) // 2]
    prefix_result, _ = replay_independently(prefix, config['entry_threshold'], config['exit_threshold'], config['initial_cash'])
    prefix_equal = replica.loc[prefix.index].equals(prefix_result)
    ref.to_csv(output / 'source_engine_daily.csv', float_format='%.15g')
    replica.to_csv(output / 'independent_replica_daily.csv', float_format='%.15g')
    save(output / 'source_engine_trades.json', original_trades)
    save(output / 'independent_replica_trades.json', trades)
    local = metrics(replica.equity)
    hk = author_hk_curve_check(tol)
    result = {'schema': 'equity-github-reference-result.v1', 'IBS': {
        'configuration': config, 'source_engine_summary': original.summary(), 'independent_replica': local,
        'same_frozen_input_engine_parity': parity, 'max_abs_error': errors,
        'trade_records_equal': equal_trades, 'causal_prefix_equal': prefix_equal,
        'closed_trades': len(original.closed_trades), 'open_positions': sum(t.is_open for t in original.trades),
        'author_headline_cagr': 0.593, 'cagr_gap_percentage_points': (local['cagr'] - 0.593) * 100,
        'within_headline_tolerance': abs(local['cagr'] - 0.593) <= tol['headline_CAGR_abs'],
        'author_original_input_snapshot_available': False, 'exact_author_window_available': False,
        'headline_cash_yield_setting_available': False, 'author_headline_fully_reproduced': False,
        'original_matching_status': 'engine_parity_only_original_data_and_settings_missing',
        'realistic_local_cost_account': None, 'optimization_allowed': False},
        'HK_published_curve_arithmetic': hk,
        'executed_financial_paths': 2, 'independent_strategy_trials': 1,
        'prior24_budget_unchanged': True, 'parameter_searches': 0, 'orders_enabled': False, 'production_eligible': False}
    save(output / 'result.json', result)
    save(DOC / 'verification.json', result)
    save(DOC / 'sources.json', receipts)
    if not parity or not prefix_equal or not all(x['arithmetic_passed'] for x in hk):
        raise AssertionError('reference checks failed; retained outputs for diagnosis')
    print(json.dumps({'IBS': result['IBS'], 'HK_checks_passed': len(hk), 'output': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
