import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
"""Part D: does any history-based strategy predict the next draw out of sample?"""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
df = pd.read_pickle('draws.pkl')
D = np.array(df.d.tolist()); n = len(D)
X = np.zeros((n, 60), dtype=np.int8)
for i, r in enumerate(D): X[i, r - 1] = 1
out = {}

# Split-half: does a number's frequency in the first half predict the second half?
h = n // 2
f1, f2 = X[:h].sum(0), X[h:].sum(0)
r, p = stats.pearsonr(f1, f2); out['split_half'] = {'r': float(r), 'p': float(p)}
# era-to-era chain
chunks = np.array_split(np.arange(n), 6)
out['era_to_next_corr'] = [float(stats.pearsonr(X[a].sum(0), X[b].sum(0))[0]) for a, b in zip(chunks[:-1], chunks[1:])]

# Walk-forward strategies: pick 6 numbers before each draw, count hits. Expected hits = 0.6
start = 300
strategies = {}
def run(pick):
    hits = []
    for t in range(start, n):
        s = pick(t); hits.append(X[t, s].sum())
    return np.array(hits)
cum = np.cumsum(X, 0)
def hot_all(t): return np.argsort(-(cum[t - 1]) + rng.random(60) * 1e-3)[:6]
def cold_all(t): return np.argsort(cum[t - 1] + rng.random(60) * 1e-3)[:6]
def hot_recent(t, w=50): return np.argsort(-X[t - w:t].sum(0) + rng.random(60) * 1e-3)[:6]
def overdue(t):
    last = np.array([np.flatnonzero(X[:t, j])[-1] for j in range(60)])
    return np.argsort(last)[:6]
def repeat_last(t): return np.flatnonzero(X[t - 1])
def random_pick(t): return rng.choice(60, 6, replace=False)
for name, f in [('hot_all_time', hot_all), ('cold_all_time', cold_all), ('hot_last_50', hot_recent),
                ('most_overdue', overdue), ('repeat_last_draw', repeat_last), ('random', random_pick)]:
    hts = run(f)
    m = hts.mean(); se = hts.std(ddof=1) / np.sqrt(len(hts))
    strategies[name] = {'mean_hits': round(float(m), 4), 'z_vs_0.6': round(float((m - 0.6) / se), 2),
                        'quadras_or_better': int((hts >= 4).sum()), 'draws': int(len(hts))}
out['strategies'] = strategies
out['expected_quadras_random'] = float((n - start) * (1 / 2332 + 1 / 154518 + 1 / 50063860))

# ML: gradient boosting on lag features, predict whether number j appears in draw t.
feats, ys, ts = [], [], []
for t in range(200, n):
    for j in range(60):
        col = X[:t, j]
        last = t - 1 - np.flatnonzero(col)[-1] if col.any() else t
        feats.append([j + 1, col.mean(), col[-10:].mean(), col[-30:].mean(), col[-100:].mean(), last, X[t - 1, j], X[t - 2, j]])
        ys.append(X[t, j]); ts.append(t)
F, Y, T = np.array(feats), np.array(ys), np.array(ts)
cut = int(n * 0.75)
tr, te = T < cut, T >= cut
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0).fit(F[tr], Y[tr])
pr = clf.predict_proba(F[te])[:, 1]
auc = roc_auc_score(Y[te], pr)
# AUC null distribution by shuffling labels within draws
null = []
Yte = Y[te].reshape(-1, 60)
for _ in range(200):
    Ys = np.array([rng.permutation(row) for row in Yte]).ravel(); null.append(roc_auc_score(Ys, pr))
# top-6 pick per draw hits
P6 = pr.reshape(-1, 60); top = np.argsort(-P6, 1)[:, :6]
hits = np.take_along_axis(Yte, top, 1).sum(1)
out['ml'] = {'train_draws': int((T[tr].max() - 199)), 'test_draws': int(len(Yte)), 'auc': round(float(auc), 4),
             'auc_null_95': [round(float(np.percentile(null, 2.5)), 4), round(float(np.percentile(null, 97.5)), 4)],
             'top6_mean_hits': round(float(hits.mean()), 4), 'expected': 0.6, 'quadras': int((hits >= 4).sum())}
json.dump(out, open('d_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
