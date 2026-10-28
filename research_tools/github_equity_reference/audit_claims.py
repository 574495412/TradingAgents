"""Build a bounded, source-bound claim catalog without treating claims as profit."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/research/equity_github_reference_20261027_v001'
DOC=ROOT/'docs/research/equities_github_reference_20261027'

# Claims deliberately preserve incompatible original accounting classes.
CLAIMS=[
('NivEyal/TQQQ-Strategy','US',.526,-.617,'backtest_margin_after_modeled_tax','period_not_given_in_headline',
 'README52.6% conflicts with committed aligned comparison47.2419931% (2015-09-11..2026-05-22) and frozen full audit46.9595735% (2011-08-04..2026-05-22). No matching headline path identified.', 'README_artifact_conflict'),
('swhy0913/Pairs-Trading-with-DTW-Clustering-and-OU-Process','HK',1.1640831551214876,-.80516082241705,'arithmetic_unnormalized_pair_PnL','247_test_rows_2018_2023_split',
 'Notebook cell14 sums pair PnLs without a funded capital denominator; annualization is mean*252, drawdown is absolute cumulative-PnL decline. Cell13 OLS uses entire supplied test series. hsci_universe.csv and V2_dtw_matrix-HK100_post_cluster.csv absent. Not account CAGR.', 'excluded_from_account_CAGR_rank'),
('Asher-S-Wu/Vectaix-Finance','HK',.3555642725,-.2009997025,'backtest_adjusted_units_after_25bps_per_side','2024-02-01..2026-09-16',
 '644 daily author NAV rows reproduce4 models arithmetic. Small LightGBM169 reference-valuation days,1 open position/1 pending sell; execution_validated=false. Training, raw prices, all scores and full trade ledger not reconstructed.', 'author_curve_arithmetic_only'),
('ZhongHong98/DCA-VS-Trend-Filtered-Strategy-Backtest-Python','US',.7666,-.8156,'contribution_contaminated_account_growth','2010-02-11..2026-02-26',
 'Monthly USD1000 contributions added to CashBook; daily return uses total portfolio value ratios; final CAGR divides by initial USD1000 without flow adjustment. Correct TWR/IRR not reconstructable from published full ledger (not supplied).', 'excluded_from_account_CAGR_rank'),
('AI4Finance-Foundation/FinRL-Trading','US',.2585,-.3895,'historical_backtest','2018-01..2025-10',
 'Separate62.16% annualized headline is5-month paper (Oct2025..Mar2026), total19.76%, DD12.22%; not live account or comparable long-history CAGR.', 'source_claim_only'),
('CazSyd/IBS-Strategy','US',.593,-.568,'backtest_TQQQ_real_listing_gross_no_trade_cost','2010-02..2026-07',
 'Patient0.132/0.965 in-sample threshold. Fresh-data original engine and independent replica match4142 bars,177 closed trades and1 open position:59.6750601% CAGR,56.8419874% DD. Original exact date/raw snapshot/headline cash yield missing;59.3% headline not fully reproduced.', 'same_input_original_engine_replica_parity'),
('HaomingJue/lev_etf_long_term_strategy','US',.348,None,'reported_expanding_walkforward_leveraged_ETF','2015..2026',
 '34.8% maxdd50 expanding walkforward claim. Full-history maxdd50 has51.8% drawdown and includes synthetic pre-inception history; do not transfer that full-history DD onto walkforward claim. No replay this round.', 'source_claim_only'),
('amit943c/MultiFactorStockRS','US',.266,-.274,'backtest_in_sample_current_stock_subset','2020-01..2025-12',
 'Top10 of~100 large-cap current subset. README explicitly in-sample, no walkforward. No replay this round.', 'source_claim_only'),
('justin10moter-ux/tqqq-trend-following','US',.336,-.549,'backtest_real_listing_TQQQ','2010..2026',
 '33.60% historical CAGR; author statement live sinceMarch2026 is not independently verified live profit. No replay this round.', 'source_claim_only'),
('davidwang0116/Quant-QQQ-QLD-TQQQ-SMA200Timing-MixedPosition-Backtest','US',.2853,-.7637,'backtest_including_synthetic_TQQQ','2000..2025',
 'Full-history28.53%, includes pre2010 synthetic3x QQQ;2016..2025 window39.28% with69.92% DD is a different window. No replay this round.', 'source_claim_only'),
]


def main():
    items=[]
    for repo,market,cagr,drawdown,kind,window,note,status in CLAIMS:
        p=DATA/repo.replace('/','__')
        m=json.loads((p/'manifest.json').read_text())
        items.append({'repo':repo,'url':f'https://github.com/{repo}/tree/{m["commit"]}',
                      'commit':m['commit'],'license':(m.get('license') or {}).get('spdx_id'),
                      'market':market,'author_claim_annual_return':cagr,'author_claim_drawdown':drawdown,
                      'accounting_class':kind,'window':window,'status':status,'finding':note,
                      'live_performance_verified':False,'production_eligible':False})
    catalog={'schema':'equity-github-claim-catalog.v1','search_date_shanghai':'2026-10-27',
             'coverage':'10 repositories from bounded public search, not all GitHub',
             'globally_highest_return_certified':False,'cross_repository_comparable_ranking_available':False,
             'US_highest_retained_long_history_real_listing_backtest_headline_in_this_set':'CazSyd/IBS-Strategy',
             'HK_highest_retained_funded_account_backtest_headline_in_this_set':'Asher-S-Wu/Vectaix-Finance',
             'selection_note':'These are source claim triage priorities after excluding non-account formulas, not verified best strategies. US short paper62.16% is a separate class; windows, risks and costs differ.',
             'repositories':items,'strategies':items,'funds':[]}
    (DOC/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    pair=DATA/'swhy0913__Pairs-Trading-with-DTW-Clustering-and-OU-Process'
    notebook=json.loads((pair/'new_pairs_trading_DTW.ipynb').read_text())
    evidence={'HK116':{'output_cell':21,'saved_output':''.join(notebook['cells'][21]['outputs'][0]['text']),
                         'annualization_formula_found':'pnl.mean()*252*100' in ''.join(notebook['cells'][14]['source']),
                         'aggregate_sum_found':"self.pnl.sum(axis = 1)" in ''.join(notebook['cells'][14]['source']),
                         'full_series_OLS_found':'sm.OLS(p0, p1).fit()' in ''.join(notebook['cells'][13]['source']),
                         'account_CAGR':None,'full_strategy_reproduced':False},
              'DCA':{'external_flows_removed':False,'corrected_TWR':None,'corrected_IRR':None},
              'NivEyal':{'README_CAGR':.526,'aligned_committed_CAGR':.4724199309615684,
                         'full_audit_committed_CAGR':.4695957348881663,'headline_parity':False}}
    dca=(DATA/'ZhongHong98__DCA-VS-Trend-Filtered-Strategy-Backtest-Python/Python/DCA_VS_Trend_Filtered_Backtest_Python.py').read_text()
    evidence['DCA']['source_lines']=[{'line':i+1,'text':s.strip()} for i,s in enumerate(dca.splitlines()) if any(x in s for x in ['AddAmount(', 'daily_return =', 'total_return =', 'cagr ='])]
    (DOC/'claim-audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('Catalog:',len(items),'repositories; no globally comparable highest-return certification')

if __name__=='__main__':main()
