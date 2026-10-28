"""Freeze public GitHub metadata and text sources; never execute downloaded code."""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/equity_github_reference_20261027_v001'
REPOS = [
    'NivEyal/TQQQ-Strategy',
    'swhy0913/Pairs-Trading-with-DTW-Clustering-and-OU-Process',
    'Asher-S-Wu/Vectaix-Finance',
    'ZhongHong98/DCA-VS-Trend-Filtered-Strategy-Backtest-Python',
    'AI4Finance-Foundation/FinRL-Trading',
    'CazSyd/IBS-Strategy',
    'HaomingJue/lev_etf_long_term_strategy',
    'amit943c/MultiFactorStockRS',
    'justin10moter-ux/tqqq-trend-following',
    'davidwang0116/Quant-QQQ-QLD-TQQQ-SMA200Timing-MixedPosition-Backtest',
]

def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'TradingAgents-public-reference-audit'}), timeout=45) as r:
        data = r.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError('source exceeds 8 MiB')
        return data

def capture(repo):
    target = OUT / repo.replace('/', '__')
    target.mkdir(parents=True, exist_ok=False)
    manifest = {'repo': repo, 'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'files': [], 'failures': []}
    def save(name, url):
        try:
            data = fetch(url)
            (target / name).write_bytes(data)
            manifest['files'].append({'path': name, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            return data
        except Exception as exc:
            manifest['failures'].append({'path': name, 'url': url, 'error': str(exc)})
    meta = save('repository.json', f'https://api.github.com/repos/{repo}')
    if meta:
        meta = json.loads(meta)
        commit = save('commit.json', f'https://api.github.com/repos/{repo}/commits/{meta["default_branch"]}')
        if commit:
            sha = json.loads(commit)['sha']
            manifest.update(commit=sha, license=meta.get('license'), stars=meta['stargazers_count'])
            tree = save('tree.json', f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1')
            if tree:
                entries = json.loads(tree)['tree']
                for entry in entries:
                    if entry['type'] == 'blob' and entry['path'].lower() in ('readme.md', 'license', 'license.md', 'license.txt'):
                        save(entry['path'], f'https://raw.githubusercontent.com/{repo}/{sha}/{entry["path"]}')
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(capture, REPOS))
    (OUT / 'manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    for item in results:
        print(item['repo'], item.get('commit'), 'license', (item.get('license') or {}).get('spdx_id'), 'failures', len(item['failures']))
