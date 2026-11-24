import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
"""Part A: is the Mega-Sena draw random? Monte Carlo tests against a perfect 6-of-60 lottery."""
import json, numpy as np, pandas as pd
from math import comb
from scipy import stats

rng = np.random.default_rng(2026)
df = pd.read_pickle('draws.pkl')
D = np.array(df.d.tolist())            # (n,6), 1..60
n = len(D)
X = np.zeros((n, 60), dtype=np.int8)
for i, row in enumerate(D):
    X[i, row - 1] = 1
out = {'n_draws': int(n), 'first': df.date.iloc[0], 'last': df.date.iloc[-1], 'last_contest': int(df.c.iloc[-1])}

def sim_draws(k):
    # k random draws, returns (k,60) indicator
    r = rng.random((k, 60)).argsort(axis=1)[:, :6]
    Y = np.zeros((k, 60), dtype=np.int8)
    np.put_along_axis(Y, r, 1, axis=1)
    return Y

# 1. Frequency uniformity
freq = X.sum(0); E = n * 6 / 60
chi = ((freq - E) ** 2 / E).sum()
NSIM = 4000
chis = np.empty(NSIM)
for s in range(NSIM):
    f = sim_draws(n).sum(0)
    chis[s] = ((f - E) ** 2 / E).sum()
out['freq'] = {'expected_each': E, 'min': int(freq.min()), 'max': int(freq.max()),
               'most': [int(i + 1) for i in np.argsort(-freq)[:6]], 'most_counts': sorted(freq.tolist(), reverse=True)[:6],
               'least': [int(i + 1) for i in np.argsort(freq)[:6]], 'least_counts': sorted(freq.tolist())[:6],
               'chi2': float(chi), 'p_mc': float((chis >= chi).mean()),
               'sim_range_typical': [float(np.percentile([0], 0))]}
# typical spread of max-min in a fair lottery
spreads = []
for s in range(1000):
    f = sim_draws(n).sum(0); spreads.append(f.max() - f.min())
out['freq']['observed_spread'] = int(freq.max() - freq.min())
out['freq']['fair_spread_median'] = float(np.median(spreads))
out['freq']['fair_spread_95'] = float(np.percentile(spreads, 95))
out['freq'].pop('sim_range_typical')
pd.DataFrame({'n': range(1, 61), 'freq': freq}).to_csv('freq.csv', index=False)

# 2. Era test: split into 6 eras, chi2 each (equipment / procedure changes)
eras = []
for chunk in np.array_split(np.arange(n), 6):
    f = X[chunk].sum(0); e = len(chunk) * 0.1
    c = ((f - e) ** 2 / e).sum()
    # MC null for this length
    cs = np.array([((sim_draws(len(chunk)).sum(0) - e) ** 2 / e).sum() for _ in range(800)])
    eras.append({'from': int(df.c.iloc[chunk[0]]), 'to': int(df.c.iloc[chunk[-1]]), 'chi2': float(c), 'p_mc': float((cs >= c).mean())})
out['eras'] = eras

# 3. Repeats from previous draw: hypergeometric(60,6,6)
rep = (X[1:] * X[:-1]).sum(1)
obs = np.bincount(rep, minlength=7)[:5]
hyp = np.array([comb(6, k) * comb(54, 6 - k) / comb(60, 6) for k in range(7)])
exp = hyp[:5] * (n - 1); exp[-1] += hyp[5:].sum() * (n - 1)
obs_c = obs.copy(); obs_c[-1] = (rep >= 4).sum()
c, p = stats.chisquare(np.r_[obs_c[:3], obs_c[3:].sum()], np.r_[exp[:3], exp[3:].sum()])
out['repeats'] = {'mean_obs': float(rep.mean()), 'mean_exp': 0.6, 'dist_obs': obs.tolist(),
                  'dist_exp': [round(x, 1) for x in (hyp * (n - 1))[:5]], 'p': float(p)}

# 4. Gaps ("atraso"): geometric with p=0.1
gaps = []
for j in range(60):
    idx = np.flatnonzero(X[:, j]); gaps += np.diff(idx).tolist()
gaps = np.array(gaps)
bins = [1, 2, 3, 5, 8, 12, 20, 30, 1000]
obsg = np.histogram(gaps, bins=bins)[0]
pg = np.array([(0.9 ** (a - 1) - 0.9 ** (b - 1)) for a, b in zip(bins[:-1], bins[1:])]); pg = pg / pg.sum()
c, p = stats.chisquare(obsg, pg * obsg.sum())
cur = [(n - 1 - np.flatnonzero(X[:, j])[-1]) for j in range(60)]
out['gaps'] = {'mean_obs': float(gaps.mean()), 'mean_exp': 10.0, 'max_gap_ever': int(gaps.max()), 'p': float(p),
               'current_most_overdue': sorted([(int(j + 1), int(cur[j])) for j in range(60)], key=lambda t: -t[1])[:5]}

# 5. Pair co-occurrence: are some pairs together too often?
P = X.T.astype(np.int32) @ X.astype(np.int32)
iu = np.triu_indices(60, 1); pairs = P[iu]
Ep = n * comb(58, 4) / comb(60, 6)
chip = ((pairs - Ep) ** 2 / Ep).sum()
chips = []
for s in range(500):
    Y = sim_draws(n).astype(np.int32); Q = (Y.T @ Y)[iu]; chips.append(((Q - Ep) ** 2 / Ep).sum())
top = np.argsort(-pairs)[:3]
out['pairs'] = {'expected_each': float(Ep), 'max_pair': [int(iu[0][top[0]] + 1), int(iu[1][top[0]] + 1), int(pairs[top[0]])],
                'chi2': float(chip), 'p_mc': float((np.array(chips) >= chip).mean()),
                'max_in_fair_lottery_median': None}
mx = [ (lambda Y: (Y.T @ Y)[iu].max())(sim_draws(n).astype(np.int32)) for _ in range(300)]
out['pairs']['max_in_fair_lottery_median'] = float(np.median(mx))

# 6. Sum of the six numbers vs exact distribution (by simulation of 2M)
sums = D.sum(1)
simsum = (rng.random((2_000_000, 60)).argsort(axis=1)[:, :6] + 1).sum(1) if False else None
# exact via DP over subsets is heavy; use 400k simulated draws
S = np.array([rng.choice(60, 6, replace=False).sum() + 6 for _ in range(1)])
sim = np.argsort(rng.random((400_000, 60)), axis=1)[:, :6].sum(1) + 6
ks = stats.ks_2samp(sums, sim)
out['sums'] = {'mean_obs': float(sums.mean()), 'mean_exp': 183.0, 'sd_obs': float(sums.std()), 'sd_sim': float(sim.std()), 'ks_p': float(ks.pvalue)}

# 7. Even/odd and low/high (<=30)
ev = (D % 2 == 0).sum(1); lo = (D <= 30).sum(1)
he = np.array([comb(30, k) * comb(30, 6 - k) / comb(60, 6) for k in range(7)])
for name, v in [('even', ev), ('low30', lo)]:
    o = np.bincount(v, minlength=7); c, p = stats.chisquare(o, he * n)
    out[name] = {'obs': o.tolist(), 'exp': [round(x, 1) for x in he * n], 'p': float(p)}

# 8. Exact repeat of a full combination ever?
combos = set(map(tuple, D)); out['repeated_full_combo'] = int(n - len(combos))

json.dump(out, open('a_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
