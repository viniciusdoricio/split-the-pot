"""Download the full Mega-Sena history (numbers, winners per tier, sales) and cross-check it.

Primary source: GabrielRossiSoares/megasena (scraped from Caixa's API, updated after each draw).
Cross-check:    sosthenis/megasena-data (independent scrape, numbers only).
Writes out/draws.pkl (used by every other script) and data/draws.csv (for humans / BI tools).
"""
import json, os, sys, urllib.request
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PRIMARY = 'https://raw.githubusercontent.com/GabrielRossiSoares/megasena/main/data.json'
CHECK = 'https://raw.githubusercontent.com/sosthenis/megasena-data/main/megasena.json'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'megasena-lab'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def main():
    g = get(PRIMARY)['results']
    rows = []
    for r in g:
        rows.append(dict(c=r['concurso'], date=r['data'], d=sorted(int(x) for x in r['dezenas']),
                         w6=r.get('ganhadores'), w5=(r.get('quina') or {}).get('ganhadores'),
                         w4=(r.get('quadra') or {}).get('ganhadores'),
                         p5=(r.get('quina') or {}).get('premio'), p4=(r.get('quadra') or {}).get('premio'),
                         sales=r.get('arrecadacao'), prize6=r.get('premio')))
    df = pd.DataFrame(rows).sort_values('c').reset_index(drop=True)

    # integrity checks
    assert df.c.is_unique, 'duplicate contests'
    gaps = (df.c.diff().dropna() != 1).sum()
    assert gaps == 0, f'{gaps} missing contests'
    assert all(len(set(x)) == 6 and min(x) >= 1 and max(x) <= 60 for x in df.d), 'bad numbers'

    try:
        s = {x['concurso']: sorted(int(y) for y in x['dezenas']) for x in get(CHECK)}
        bad = [c for c, d in zip(df.c, df.d) if c in s and s[c] != d]
        print(f'cross-check vs second scrape: {len(s)} contests compared, {len(bad)} mismatches {bad[:10]}')
        if bad:
            sys.exit('mismatch between sources: investigate before analysing')
    except Exception as e:  # cross-check source down is not fatal
        print('cross-check skipped:', e)

    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    df.to_pickle(os.path.join(ROOT, 'out', 'draws.pkl'))
    flat = df.copy(); flat[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']] = pd.DataFrame(flat.d.tolist())
    flat.drop(columns='d').to_csv(os.path.join(ROOT, 'data', 'draws.csv'), index=False)
    print(f'{len(df)} draws, contest {df.c.min()}..{df.c.max()} (last {df.date.iloc[-1]}), '
          f'{(df.sales > 0).sum()} with sales data')


if __name__ == '__main__':
    main()
