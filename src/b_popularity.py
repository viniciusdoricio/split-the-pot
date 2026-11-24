import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
"""Part B: infer which numbers/patterns Brazilian bettors pick, from how many people won quadra/quina.
Model (Farrell-Walker style): W4_d ~ quasi-Poisson, log E[W4_d] = log(sales_d) + period FE + sum_i beta_i*[i in draw d] (+ pattern terms)
If a drawn combination is made of popular numbers, many more people share its 4-subsets -> more quadra winners than sales explain."""
import json, numpy as np, pandas as pd
import statsmodels.api as sm

df = pd.read_pickle('draws.pkl')
df['dt'] = pd.to_datetime(df.date, dayfirst=True)
d = df[(df.sales > 0) & (df.w4 > 0)].copy().reset_index(drop=True)
D = np.array(d.d.tolist())
d['virada'] = ((d.dt.dt.month == 12) & (d.dt.dt.day == 31)) | ((d.dt.dt.month == 1) & (d.dt.dt.day == 1) & (d.dt.dt.year == 2026))
# Virada contest numbers are flagged by date; also by sales outlier
d['virada'] = d['virada'] | (d.sales > 8 * d.sales.rolling(9, center=True, min_periods=1).median())
period = d.dt.dt.to_period('Q').astype(str)
FE = pd.get_dummies(period, drop_first=True, dtype=float)
FE['virada'] = d.virada.astype(float)
# number indicators, sum-to-zero coding (effects relative to the average number)
N = np.zeros((len(d), 60))
for i, r in enumerate(D): N[i, r - 1] = 1
Nc = N[:, :59] - N[:, [59]]          # effect of 60 = -sum(others)
Xmat = np.column_stack([np.ones(len(d)), FE.values, Nc])
offset = np.log(d.sales.values)
out = {'draws_used': int(len(d)), 'from': d.date.iloc[0], 'to': d.date.iloc[-1], 'virada_draws': int(d.virada.sum())}

def fit(y, X):
    m = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(scale='X2')
    return m
res = {}
for tier in ['w4', 'w5']:
    m = fit(d[tier].values, Xmat)
    b = m.params[-59:]; se = m.bse[-59:]
    b60 = -b.sum()
    beta = np.r_[b, b60]
    # SE for 60 via covariance
    C = m.cov_params()[-59:, -59:]; se60 = np.sqrt(np.ones(59) @ C @ np.ones(59))
    ses = np.r_[se, se60]
    res[tier] = {'beta': beta, 'se': ses, 'dispersion': float(m.scale)}
    out[f'{tier}_dispersion'] = float(m.scale)
# Number popularity per number: in quadra model, a number's beta reflects popularity of 4-subsets containing it.
b4, s4 = res['w4']['beta'], res['w4']['se']
b5 = res['w5']['beta']
pop = pd.DataFrame({'n': range(1, 61), 'beta_quadra': b4, 'se': s4, 'beta_quina': b5,
                    'pct_effect': (np.exp(b4) - 1) * 100})
pop.to_csv('popularity.csv', index=False)
out['corr_quadra_quina_betas'] = float(np.corrcoef(b4, b5)[0, 1])
out['most_popular'] = pop.sort_values('beta_quadra', ascending=False).head(10)[['n', 'pct_effect']].round(1).values.tolist()
out['least_popular'] = pop.sort_values('beta_quadra').head(10)[['n', 'pct_effect']].round(1).values.tolist()
out['mean_effect_1_31'] = float(pop.pct_effect[:31].mean()); out['mean_effect_32_60'] = float(pop.pct_effect[31:].mean())

# Pattern model: interpretable features instead of 60 dummies
def feats(r):
    r = np.sort(r)
    return {'n_le31': int((r <= 31).sum()), 'n_le12': int((r <= 12).sum()),
            'consec_pairs': int((np.diff(r) == 1).sum()),
            'same_col_max': int(np.bincount(r % 10, minlength=10).max()),       # volante columns = last digit
            'same_row_max': int(np.bincount((r - 1) // 10, minlength=6).max()), # volante rows = decades
            'mult5': int((r % 5 == 0).sum()), 'has_lucky': int(len(set(r) & {7, 13, 10, 3, 5, 9}) )}
F = pd.DataFrame([feats(r) for r in D])
Xp = np.column_stack([np.ones(len(d)), FE.values, F.values])
mp = sm.GLM(d.w4.values, Xp, family=sm.families.Poisson(), offset=offset).fit(scale='X2')
k = F.shape[1]
out['pattern_model_quadra'] = {c: {'pct_per_unit': round(float((np.exp(mp.params[-k + i]) - 1) * 100), 2),
                                   'z': round(float(mp.params[-k + i] / mp.bse[-k + i]), 1)} for i, c in enumerate(F.columns)}
mp5 = sm.GLM(d.w5.values, Xp, family=sm.families.Poisson(), offset=offset).fit(scale='X2')
out['pattern_model_quina'] = {c: {'pct_per_unit': round(float((np.exp(mp5.params[-k + i]) - 1) * 100), 2),
                                  'z': round(float(mp5.params[-k + i] / mp5.bse[-k + i]), 1)} for i, c in enumerate(F.columns)}

# Out-of-sample check: fit number betas on draws before 2020, predict quadra residual ratio after 2020
cut = d.dt < '2020-01-01'
def fit_sub(mask, y):
    Xs = Xmat[mask]; keep = Xs[:, 1:FE.shape[1] + 1].std(0) > 0
    cols = np.r_[True, keep, np.ones(59, bool)]
    return sm.GLM(y[mask], Xs[:, cols], family=sm.families.Poisson(), offset=offset[mask]).fit(scale='X2')
m_early = fit_sub(cut.values, d.w4.values); m_late = fit_sub(~cut.values, d.w4.values)
be, bl = m_early.params[-59:], m_late.params[-59:]
be, bl = np.r_[be, -be.sum()], np.r_[bl, -bl.sum()]
out['popularity_stability_corr_pre2020_vs_post2020'] = float(np.corrcoef(be, bl)[0, 1])

json.dump(out, open('b_results.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
