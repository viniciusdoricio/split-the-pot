"""ROADMAP 1: estimate the random-bet floor f from quadra/quina winner counts, validate on jackpot winners.

Model and identification: see src/floor_model.py. Pre-registered plan, decision rules and the count of tries:
README "Random-bet floor" section. Run src/f_floor_sim.py first: it says whether f is recoverable at all.

Steps
 0. sanity: the log-linear model in this framework reproduces b_popularity.py; structural theta at f=0 tracks beta
 1. primary: joint W4+W5 fit, profile likelihood for f (grid 0.02), CI from the profile, plus a calendar-year bootstrap
 2. diagnostics (Bonferroni over 4): W4-only, W5-only, before/after 2015
 3. validation: 10 contiguous blocks, popularity fitted on the other 9, jackpot (W6) log score vs the current model
 4. reais: Virada EV rows at f-hat and at both CI ends, plus EV as a function of f
Writes out/f_results.json.
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v, '1')
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from math import comb
import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
import floor_model as fm
import floor_data
from common import OUT, C60, ev

GRID_FINE = np.round(np.arange(0, 0.901, 0.02), 4)
GRID_FAST = np.round(np.arange(0, 0.901, 0.05), 4)
CRIT_95 = chi2.ppf(0.95, 1)
CRIT_BONF4 = chi2.ppf(1 - 0.05 / 4, 1)
G = {}


def mult(theta, f, M, k):
    return f + (1 - f) * fm.struct_Q(theta, M, k) / fm.P[k]


# ---------- 1. bootstrap worker ----------
def _init_boot():
    G['d'], G['data'] = floor_data.load()


def _boot(seed):
    d, data = G['d'], G['data']
    rng = np.random.default_rng(seed)
    years = d.dt.dt.year.values; uy = np.unique(years)
    pick = rng.choice(uy, len(uy), replace=True)
    idx = np.concatenate([np.flatnonzero(years == y) for y in pick])
    # duplicated years get their own quarter groups so each copy keeps its own effect
    sub = data.subset(idx)
    copy_id = np.concatenate([np.full((years == y).sum(), i) for i, y in enumerate(pick)])
    _, sub.g = np.unique(np.c_[data.g[idx], copy_id], axis=0, return_inverse=True); sub.g = sub.g.ravel(); sub.ng = int(sub.g.max()) + 1
    return fm.fit_struct(sub, grid=GRID_FAST, ci=False)['f']


# ---------- 3. cross-fit worker ----------
def _fold(args):
    b, train_idx, test_idx = args
    d, data = G['d'], G['data']
    y4, y6, vir = data.tiers[4], d.w6.values.astype(float), data.vir
    tr, te = data.subset(train_idx), data.subset(test_idx)
    res = {}
    # current model: log-linear beta on W4, jackpot multiplier exp(1.5*sum beta), normalised to average 1 over tickets
    beta, v4 = fm.fit_loglin(tr, 4)
    norm = fm.esp(np.exp(1.5 * beta), np.ones((1, 60)))[0, 6] / C60
    for name, D_ in (('train', tr), ('test', te)):
        s = D_.M @ beta
        res[f'cur_off_{name}'] = fm.w4_exposure(D_, D_.tiers[4], v4, np.exp(s))
        res[f'cur_s_{name}'] = s
        res[f'cur_lm6_{name}'] = 1.5 * s - np.log(norm)
    # new model: structural theta + f on W4+W5
    r = fm.fit_struct(tr, grid=GRID_FAST, ci=False)
    for name, D_ in (('train', tr), ('test', te)):
        res[f'new_off_{name}'] = fm.w4_exposure(D_, D_.tiers[4], r['v'][4], mult(r['theta'], r['f'], D_.M, 4))
        res[f'new_lm6_{name}'] = np.log(mult(r['theta'], r['f'], D_.M, 6))
    y6tr, y6te = y6[train_idx], y6[test_idx]
    out = {'block': b, 'f_train': r['f'], 'test_idx': test_idx}
    p = fm.w6_calibrate(y6tr, res['cur_off_train'] + res['cur_lm6_train'], tr.vir)
    out['mu_cur'] = fm.w6_mu(p, res['cur_off_test'] + res['cur_lm6_test'], te.vir)
    p = fm.w6_calibrate(y6tr, res['cur_off_train'], tr.vir, extra=res['cur_s_train'])
    out['mu_curk'] = fm.w6_mu(p, res['cur_off_test'], te.vir, extra=res['cur_s_test']); out['kappa_train'] = float(p[2])
    p = fm.w6_calibrate(y6tr, res['new_off_train'] + res['new_lm6_train'], tr.vir)
    out['mu_new'] = fm.w6_mu(p, res['new_off_test'] + res['new_lm6_test'], te.vir)
    out['m6_new'] = np.exp(res['new_lm6_test'])
    return out


def year_boot_ci(diff, years, reps=2000, seed=5):
    rng = np.random.default_rng(seed); uy = np.unique(years)
    by = {y: diff[years == y].sum() for y in uy}; vals = np.array([by[y] for y in uy])
    sums = np.array([vals[rng.integers(0, len(uy), len(uy))].sum() for _ in range(reps)])
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


# ---------- 4. reais ----------
def ev_rows(theta, f, n_tickets=400_000, seed=31):
    rng = np.random.default_rng(seed)
    R = np.argsort(rng.random((n_tickets, 60)), 1)[:, :6]
    T = np.zeros((n_tickets, 60)); np.put_along_axis(T, R, 1, axis=1)
    m6 = mult(theta, f, T, 6)
    rows = {}
    for label, q in [('popular_p95', 95), ('median_p50', 50), ('unpopular_p5', 5), ('unpopular_p1', 1)]:
        i = int(np.argmin(np.abs(m6 - np.percentile(m6, q))))
        Mc, M5, M4 = m6[i], mult(theta, f, T[i:i + 1], 5)[0], mult(theta, f, T[i:i + 1], 4)[0]
        rows[label] = {'ticket_example': (np.flatnonzero(T[i]) + 1).tolist(), 'M6': round(float(Mc), 3),
                       **{lab: ev(Mc, B, M5=M5, M4=M4) for lab, B in [('B_520M', 520e6), ('B_620M', 620e6)]}}
    rows['average'] = {'M6': 1.0, **{lab: ev(1.0, B) for lab, B in [('B_520M', 520e6), ('B_620M', 620e6)]}}
    rows['mean_M6_check'] = round(float(m6.mean()), 4)
    return rows


def main(boot_reps, workers):
    t0 = time.time()
    d, data = floor_data.load()
    out = {'draws': int(len(d)), 'tiers_fitted': [4, 5], 'w6_used_in_fitting': False}

    # 0. sanity
    b_old = pd.read_csv(os.path.join(OUT, 'popularity.csv')).beta_quadra.values
    d4 = floor_data.load(tiers=(4,))[1]
    beta, _ = fm.fit_loglin(d4, 4)
    r0 = fm.fit_struct(d4, f_fixed=0.0)
    sl = float(np.polyfit(r0['theta'], b_old, 1)[0])
    out['sanity'] = {'loglin_vs_b_popularity_max_abs_diff': float(np.abs(beta - b_old).max()),
                     'struct_theta_f0_vs_beta_r': float(np.corrcoef(r0['theta'], b_old)[0, 1]),
                     'slope_beta_on_theta': sl, 'first_order_theory_slope': 4 / 6 - 2 / 54,
                     'implied_jackpot_exponent_beta_scale': 1 / sl}

    # 1. primary
    best = fm.fit_struct(data, grid=GRID_FINE, crit=CRIT_95)
    phi_at_best = {k: fm.pearson_phi(y, data.se(best['v'][k]), data.g, best['f'] * fm.P[k] + (1 - best['f']) * fm.struct_Q(best['theta'], data.M, k), data.ng, fm.N_PAR)
                   for k, y in data.tiers.items()}
    with ProcessPoolExecutor(workers, initializer=_init_boot) as ex:
        boots = np.array(list(ex.map(_boot, range(1000, 1000 + boot_reps))))
    boot_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    prof_ci = list(best['ci'])
    wider = boot_ci if (boot_ci[1] - boot_ci[0]) > (prof_ci[1] - prof_ci[0]) else prof_ci
    out['primary'] = {'f_hat': best['f'], 'profile_ci95': prof_ci, 'ci_open_low': best['ci_open_low'], 'ci_open_high': best['ci_open_high'],
                      'D_at_f0': best['D_at_0'], 'bootstrap_ci95_by_year': boot_ci, 'bootstrap_sd': float(boots.std(ddof=1)),
                      'bootstrap_reps': boot_reps, 'reported_ci95': wider, 'phi_used': best['phi'], 'phi_at_f_hat': phi_at_best,
                      'profile': {'f': best['profile_f'], 'D': [round(x, 3) for x in best['profile_D']]},
                      'virada_offset': best['v']}

    # 2. diagnostics, Bonferroni over 4
    diag = {}
    years = d.dt.dt.year.values
    for name, sub in [('W4_only', floor_data.load(tiers=(4,))[1]), ('W5_only', floor_data.load(tiers=(5,))[1]),
                      ('before_2015', data.subset(np.flatnonzero(years < 2015))), ('2015_on', data.subset(np.flatnonzero(years >= 2015)))]:
        r = fm.fit_struct(sub, grid=GRID_FINE, crit=CRIT_BONF4)
        diag[name] = {'f_hat': r['f'], 'ci_bonferroni4': list(r['ci']), 'ci_open_high': r['ci_open_high'], 'draws': int(len(sub.s))}
    def overlap(a, b): return not (a[1] < b[0] or b[1] < a[0])
    diag['W4_W5_cis_overlap'] = overlap(diag['W4_only']['ci_bonferroni4'], diag['W5_only']['ci_bonferroni4'])
    diag['eras_cis_overlap'] = overlap(diag['before_2015']['ci_bonferroni4'], diag['2015_on']['ci_bonferroni4'])
    out['diagnostics'] = diag

    # 3. cross-fitted jackpot validation
    blocks = np.array_split(np.arange(len(d)), 10)
    tasks = [(b, np.setdiff1d(np.arange(len(d)), blk), blk) for b, blk in enumerate(blocks)]
    with ProcessPoolExecutor(workers, initializer=_init_boot) as ex:
        folds = list(ex.map(_fold, tasks))
    y6 = d.w6.values.astype(float)
    mu = {k: np.empty(len(d)) for k in ('cur', 'curk', 'new')}; m6 = np.empty(len(d))
    for fo in folds:
        for k in mu: mu[k][fo['test_idx']] = fo[f'mu_{k}']
        m6[fo['test_idx']] = fo['m6_new']
    ls = {k: fm.pois_logpmf(y6, mu[k]) for k in mu}
    delta = ls['new'] - ls['cur']; delta_k = ls['new'] - ls['curk']
    D_tot = float(delta.sum()); ci = year_boot_ci(delta, years)
    verdict = 'pass (better)' if D_tot >= 0 else ('pass (non-inferior)' if (D_tot > -3 and ci[0] <= 0 <= ci[1]) else 'fail')
    dec = pd.qcut(m6, 10, labels=False)
    calib = [{'decile': int(q), 'mean_m6': float(m6[dec == q].mean()), 'obs_w6': int(y6[dec == q].sum()),
              'exp_new': round(float(mu['new'][dec == q].sum()), 1), 'exp_cur': round(float(mu['cur'][dec == q].sum()), 1)} for q in range(10)]
    vir = d.virada.values
    i2110 = np.flatnonzero(d.c.values == 2110)
    nb = int((delta > 0).sum()); nn = int((delta != 0).sum())
    out['validation_w6'] = {
        'blocks': 10, 'total_w6': int(y6.sum()),
        'logscore': {k: float(v.sum()) for k, v in ls.items()},
        'delta_new_minus_current_k1.5': D_tot, 'delta_ci95_year_bootstrap': ci,
        'delta_new_minus_current_kfitted': float(delta_k.sum()), 'delta_kfitted_ci95': year_boot_ci(delta_k, years),
        'verdict_preregistered': verdict,
        'robustness': {'delta_viradas_only': float(delta[vir].sum()), 'delta_excl_viradas': float(delta[~vir].sum()),
                       'delta_excl_viradas_ci95': year_boot_ci(delta[~vir], years[~vir]),
                       'delta_excl_contest_2110': float(delta.sum() - delta[i2110].sum()),
                       'draws_new_better': nb, 'draws_compared': nn, 'sign_test_p': float(binomtest(nb, nn).pvalue),
                       'delta_by_block': [round(float(delta[b].sum()), 2) for b in blocks]},
        'f_by_fold': [round(fo['f_train'], 3) for fo in folds], 'kappa_fitted_by_fold': [round(fo['kappa_train'], 3) for fo in folds],
        'calibration_by_decile_of_new_m6': calib,
        'virada_draws': [{'contest': int(c), 'w6': int(w), 'mu_new': round(float(a), 2), 'mu_cur': round(float(b), 2)}
                         for c, w, a, b in zip(d.c[vir], y6[vir], mu['new'][vir], mu['cur'][vir])]}

    # 4. reais
    fh = best['f']; lo, hi = wider
    rows = {'f_hat': {'f': fh, **ev_rows(best['theta'], fh)}}
    for lab, f in [('ci_low', lo), ('ci_high', hi)]:
        r = fm.fit_fixed(data, f, best['phi'], best['x'])
        rows[lab] = {'f': f, **ev_rows(r['theta'], f)}
    curve = []
    for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r = fm.fit_fixed(data, f, best['phi'], best['x'])
        e = ev_rows(r['theta'], f)
        curve.append({'f': f, 'D': round(2 * (best['loglik'] - r['loglik']), 2),
                      'p1_M6': e['unpopular_p1']['M6'], 'p1_mu_cowinners_520M': e['unpopular_p1']['B_520M']['mu_cowinners'],
                      'p1_ev_520M': e['unpopular_p1']['B_520M']['ev_total_per_R$6'], 'p5_ev_520M': e['unpopular_p5']['B_520M']['ev_total_per_R$6'],
                      'p1_ev_620M': e['unpopular_p1']['B_620M']['ev_total_per_R$6']})
    out['virada_2026_ev'] = rows
    out['ev_curve_over_f'] = curve

    # pre-registered stop rules (any one -> f is not identified as the share of random bets)
    rules = {}
    sim_path = os.path.join(OUT, 'f_sim.json')
    if os.path.exists(sim_path):
        S = json.load(open(sim_path))['summary']; s25 = S['f=0.25']; mis = S['misspec f=0 + patterns']
        rules['1_sim_recovery_ok'] = bool(s25['mean_ci_width'] <= 0.30 and s25['coverage'] >= 0.85 and abs(s25['bias']) <= 0.05)
        rules['2_misspec_no_spurious_floor'] = bool(mis['share_f_hat_gt_0.10'] <= 0.5)
    lo_r, hi_r = out['primary']['reported_ci95']
    covered = max(0.0, min(hi_r, 0.4) - max(lo_r, 0.0)) / 0.4
    rules['3_real_ci_informative'] = bool(covered <= 0.5 and not (out['primary']['f_hat'] <= 1e-6 and hi_r > 0.4))
    rules['4_W4_W5_agree'] = bool(diag['W4_W5_cis_overlap'])
    out['stop_rules'] = {**rules, 'f_identified': all(rules.values())}
    out['runtime_min'] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(os.path.join(OUT, 'f_results.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=200); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    o = main(a.boot, a.workers)
    print(json.dumps({k: o[k] for k in ('sanity', 'primary', 'diagnostics')}, indent=1, default=float))
    v = o['validation_w6']; print('W6 validation:', {k: v[k] for k in ('logscore', 'delta_new_minus_current_k1.5', 'delta_ci95_year_bootstrap', 'verdict_preregistered')})
    print('EV curve:', json.dumps(o['ev_curve_over_f'], indent=0))
