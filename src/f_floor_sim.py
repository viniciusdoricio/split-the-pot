"""Can the random-bet floor f be recovered from quadra/quina counts at all? (run BEFORE reading the real-data f)

Synthetic data on the real draws, sales, quarters and Virada flags, generated from the structural model fitted
with f held at each true value, with lognormal noise sized to the observed quasi-Poisson dispersion per tier.
Scenarios: true f in {0, 0.10, 0.25, 0.40}, plus a misspecification case (true f = 0 plus combination-pattern
effects that the structural model doesn't contain). Each rep re-runs the full estimation (profile + CI).
Also measures how much the W6 (jackpot) log score can tell f apart: fitted-f model vs f=0 model.

Writes out/f_sim.json. Usage: python src/f_floor_sim.py [--reps 50] [--workers 8]
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v, '1')
import argparse, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import floor_model as fm
import floor_data
from common import OUT

GRID = np.round(np.arange(0, 0.901, 0.05), 4)
TRUE_F = [0.0, 0.10, 0.25, 0.40]
PATTERN_FEATS = ['consec_pairs', 'same_col_max', 'same_row_max', 'mult5']   # combination shapes, not number-level
G = {}


def pattern_features(D):
    D = np.sort(D, 1)
    return np.column_stack([(np.diff(D, axis=1) == 1).sum(1),
                            np.array([np.bincount(r % 10, minlength=10).max() for r in D]),
                            np.array([np.bincount((r - 1) // 10, minlength=6).max() for r in D]),
                            (D % 5 == 0).sum(1)])


def setup():
    """Generating parameters per scenario, from the real data with f held at the true value (f is never estimated here)."""
    d, data = floor_data.load()
    y6 = d.w6.values.astype(float)
    D = np.array(d.d.tolist())
    b = json.load(open(os.path.join(OUT, 'b_results.json')))
    F = pattern_features(D)
    g4 = np.array([np.log1p(b['pattern_model_quadra'][c]['pct_per_unit'] / 100) for c in PATTERN_FEATS])
    g5 = np.array([np.log1p(b['pattern_model_quina'][c]['pct_per_unit'] / 100) for c in PATTERN_FEATS])
    pat = {4: np.exp(F @ g4), 5: np.exp(F @ g5), 6: np.exp(1.5 * (F @ g4))}
    scen = {}
    for name, f, use_pat in [(f'f={f:.2f}', f, False) for f in TRUE_F] + [('misspec f=0 + patterns', 0.0, True)]:
        phi, r = fm.estimate_phi(data, f)
        t, v = r['theta'], r['v']
        mu = {}
        for k in (4, 5):
            m = (f * fm.P[k] + (1 - f) * fm.struct_Q(t, data.M, k)) * (pat[k] if use_pat else 1)
            se = data.se(v[k]); mu[k] = se * fm.alphas(data.tiers[k], se, data.g, m, data.ng)[data.g] * m
        m4 = f + (1 - f) * fm.struct_Q(t, data.M, 4) / fm.P[4]
        m6 = (f + (1 - f) * fm.struct_Q(t, data.M, 6) / fm.P[6]) * (pat[6] if use_pat else 1)
        off = fm.w4_exposure(data, data.tiers[4], v[4], m4) + np.log(m6)
        cal = fm.w6_calibrate(y6, off, data.vir)
        mu[6] = fm.w6_mu(cal, off, data.vir)
        scen[name] = {'f': f, 'phi': phi, 'mu': mu}
    return data, scen


def init():
    G['data'], G['scen'] = setup()


def one(args):
    name, rep = args
    data, sc = G['data'], G['scen'][name]
    rng = np.random.default_rng([abs(hash(name)) % 2**31, rep])
    y = {}
    for k in (4, 5):
        mu = sc['mu'][k]; s2 = np.log1p(np.maximum(sc['phi'][k] - 1, 0) / mu)
        y[k] = rng.poisson(mu * np.exp(rng.normal(0, np.sqrt(s2)) - s2 / 2)).astype(float)
    y6 = rng.poisson(sc['mu'][6]).astype(float)
    syn = fm.Data(data.M, data.s, data.g, y, data.vir)
    res = fm.fit_struct(syn, grid=GRID)
    r0 = fm.fit_fixed(syn, 0.0, res['phi'], res['x'])
    def score(r):
        f = r['f']; t = r['theta']
        m4 = f + (1 - f) * fm.struct_Q(t, syn.M, 4) / fm.P[4]
        off = fm.w4_exposure(syn, y[4], r['v'][4], m4) + np.log(f + (1 - f) * fm.struct_Q(t, syn.M, 6) / fm.P[6])
        return float(fm.pois_logpmf(y6, fm.w6_mu(fm.w6_calibrate(y6, off, syn.vir), off, syn.vir)).sum())
    lo, hi = res['ci']
    return {'scenario': name, 'rep': rep, 'f_hat': res['f'], 'ci_lo': lo, 'ci_hi': hi,
            'ci_open_high': res['ci_open_high'], 'phi4': res['phi'][4], 'phi5': res['phi'][5],
            'dscore_w6_fhat_vs_f0': score(res) - score(r0)}


def summarise(rows, scen):
    out = {}
    for name, sc in scen.items():
        R = [r for r in rows if r['scenario'] == name]
        f = sc['f']; fh = np.array([r['f_hat'] for r in R])
        lo = np.array([r['ci_lo'] for r in R]); hi = np.array([r['ci_hi'] for r in R])
        ds = np.array([r['dscore_w6_fhat_vs_f0'] for r in R])
        out[name] = {'true_f': f, 'reps': len(R), 'mean_f_hat': float(fh.mean()), 'bias': float(fh.mean() - f),
                     'rmse': float(np.sqrt(((fh - f) ** 2).mean())), 'sd_f_hat': float(fh.std(ddof=1)),
                     'mean_ci_width': float((hi - lo).mean()), 'coverage': float(((lo <= f) & (f <= hi)).mean()),
                     'share_ci_open_high': float(np.mean([r['ci_open_high'] for r in R])),
                     'share_f_hat_gt_0.10': float((fh > 0.10).mean()),
                     'w6_logscore_gain_fhat_vs_f0_mean': float(ds.mean()), 'w6_logscore_gain_sd': float(ds.std(ddof=1)),
                     'target_phi': {k: round(v, 1) for k, v in sc['phi'].items()},
                     'simulated_phi_mean': {4: float(np.mean([r['phi4'] for r in R])), 5: float(np.mean([r['phi5'] for r in R]))}}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=50); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    data, scen = setup()
    tasks = [(name, r) for name in scen for r in range(a.reps)]
    with ProcessPoolExecutor(a.workers, initializer=init) as ex:
        rows = list(ex.map(one, tasks, chunksize=2))
    res = {'reps_per_scenario': a.reps, 'grid': GRID.tolist(), 'summary': summarise(rows, scen), 'rows': rows,
           'runtime_min': round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(OUT, 'f_sim.json'), 'w'), indent=1)
    print(json.dumps(res['summary'], indent=1))
