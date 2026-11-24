import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
"""Part E: generate low-popularity Mega da Virada tickets and export model data for the web tool."""
import json, numpy as np, pandas as pd
from math import comb

rng = np.random.default_rng()
df = pd.read_pickle('draws.pkl')
pop = pd.read_csv('popularity.csv'); b4 = pop.beta_quadra.values
past = set(map(tuple, df.d))
R = np.argsort(np.random.default_rng(1).random((400_000, 60)), 1)[:, :6]
sr = b4[R].sum(1); norm = float(np.exp(1.5 * sr).mean())
M = lambda c: float(np.exp(1.5 * b4[np.asarray(c) - 1].sum()) / norm)

def ok(c):
    c = np.sort(c)
    if tuple(c) in past: return False
    if (np.diff(c) == 1).sum() > 1: return False                       # sequences are played by many
    if np.bincount(c % 10, minlength=10).max() > 2: return False        # volante column lines
    if np.bincount((c - 1) // 10, minlength=6).max() > 3: return False  # volante row lines
    d = np.diff(c)
    if any(d[i] == d[i + 1] == d[i + 2] for i in range(3)): return False  # arithmetic runs
    if (c % 5 == 0).sum() > 2: return False
    return True

def generate(k=10, pct=5, max_overlap=2, draws=300_000):
    C = np.sort(np.argsort(rng.random((draws, 60)), 1)[:, :6] + 1, 1)
    s = b4[C - 1].sum(1); thr = np.percentile(s, pct)
    pool = C[s <= thr]; rng.shuffle(pool)        # random among the cheap ones: avoid the single "optimal" ticket others may also find
    picked = []
    for c in pool:
        if not ok(c): continue
        if all(len(set(c) & set(p)) <= max_overlap for p in picked):
            picked.append(c)
        if len(picked) == k: break
    return [(list(map(int, c)), round(M(c), 2)) for c in picked]

tickets = generate()
print(json.dumps(tickets))
json.dump({'tickets': tickets}, open('e_tickets.json', 'w'))
# export for the web tool
json.dump({'beta': [round(float(x), 5) for x in b4], 'norm': norm,
           'past': [list(map(int, c)) for c in df.d], 'last_contest': int(df.c.max()), 'last_date': df.date.iloc[-1]},
          open('model_export.json', 'w'))
