import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
"""Part C: (1) validate popularity on actual SENA winner counts, (2) Mega da Virada 2026 expected value by combo type,
(3) build an optimizer for low-popularity combinations."""
import json, numpy as np, pandas as pd
import statsmodels.api as sm
from math import comb

rng = np.random.default_rng(31)
df = pd.read_pickle('draws.pkl'); df['dt'] = pd.to_datetime(df.date, dayfirst=True)
pop = pd.read_csv('popularity.csv')
b4 = pop.beta_quadra.values; b5 = pop.beta_quina.values
out = {}
# ratio quina/quadra betas -> exponent check (theory for independent-weights model: 5/4 = 1.25)
slope = np.polyfit(b4, b5, 1)[0]; out['slope_b5_on_b4'] = float(slope)

def score(c):  # sum of quadra betas over the 6 numbers
    return b4[np.asarray(c) - 1].sum()

# (1) Sena validation: w6 ~ Poisson, log E = log(sales) + year FE + virada + kappa*score
d = df[(df.sales > 0)].copy().reset_index(drop=True)
d['s'] = [score(c) for c in d.d]
d['virada'] = d.sales > 8 * d.sales.rolling(9, center=True, min_periods=1).median()
FE = pd.get_dummies(d.dt.dt.year.astype(str), drop_first=True, dtype=float); FE['virada'] = d.virada.astype(float)
X = np.column_stack([np.ones(len(d)), FE.values, d.s.values])
m = sm.GLM(d.w6.values, X, family=sm.families.Poisson(), offset=np.log(d.sales.values)).fit()
out['sena_validation'] = {'kappa': float(m.params[-1]), 'se': float(m.bse[-1]), 'z': float(m.params[-1] / m.bse[-1]),
                          'theory_kappa_6over4': 1.5, 'total_sena_winners': int(d.w6.sum()), 'draws': int(len(d))}
kappa = 1.5  # use theory value; report the fitted one alongside

# Popularity multiplier for a 6-combo, normalised so a uniformly random combo averages 1
R = np.argsort(rng.random((400_000, 60)), 1)[:, :6]
sr = b4[R].sum(1)
norm = np.exp(kappa * sr).mean()
def M(c, k=kappa): return float(np.exp(k * score(c)) / np.exp(k * sr).mean()) if k != kappa else float(np.exp(kappa * score(c)) / norm)
pct = {q: float(np.exp(kappa * np.percentile(sr, q)) / norm) for q in [1, 5, 25, 50, 75, 95, 99]}
out['M_distribution_random_combos'] = pct

# (2) Virada 2025 back-check
v25 = df[df.c == 2955].iloc[0]
B25 = v25.sales / 6.0
out['virada_2025'] = {'numbers': list(map(int, v25.d)), 'sales': float(v25.sales), 'combos_bought': float(B25),
                      'M_model': M(v25.d), 'expected_sena_winners_avg_combo': B25 / comb(60, 6),
                      'expected_sena_winners_this_combo': B25 / comb(60, 6) * M(v25.d), 'actual_sena_winners': int(v25.w6),
                      'expected_quadra_avg': B25 / 2332, 'actual_quadra': int(v25.w4),
                      'implied_M4_from_quadra': float(v25.w4 / (B25 / 2332))}

# (3) Virada 2026 EV per R$6 ticket
def ev(Mc, B, sena_pool=1.0e9):
    mu = B / comb(60, 6) * Mc
    share = (1 - np.exp(-mu)) / mu                       # E[1/(1+K)], K~Poisson(mu)
    jack = sena_pool / comb(60, 6) * share
    total_rateio = sena_pool / 0.90
    q5 = 0.05 * total_rateio; q4 = 0.05 * total_rateio
    # quina/quadra: thousands of winners, share ~ pool / expected winners (popularity scales ~M^(5/6), M^(4/6))
    e5 = q5 / max(B / 154_518 * Mc ** (5 / 6), 1) / 154_518
    e4 = q4 / max(B / 2_332 * Mc ** (4 / 6), 1) / 2_332
    return {'mu_cowinners': round(mu, 2), 'expected_share_of_jackpot': round(share, 4),
            'jackpot_if_you_win_expected_R$': round(sena_pool * share), 'ev_jackpot': round(jack, 2),
            'ev_quina': round(e5, 3), 'ev_quadra': round(e4, 3), 'ev_total_per_R$6': round(jack + e5 + e4, 2),
            'return_pct': round((jack + e5 + e4) / 6 * 100, 1)}
scen = {}
for label, B in [('B_520M (≈2025 sales)', 520e6), ('B_620M (+20%)', 620e6)]:
    scen[label] = {'popular_combo_p95': ev(pct[95], B), 'average_combo': ev(1.0, B), 'median_combo': ev(pct[50], B),
                   'unpopular_combo_p5': ev(pct[5], B), 'unpopular_combo_p1': ev(pct[1], B)}
# conservative: assume 25% of all bets are uniform random (Surpresinha), which floors popularity
f = 0.25
scen['conservative_floor_25pct_random_B_520M'] = {'unpopular_p1': ev(f + (1 - f) * pct[1], 520e6), 'average': ev(1.0, 520e6)}
out['virada_2026_ev'] = scen
# classic popular pattern examples
ex = {'1-2-3-4-5-6': [1, 2, 3, 4, 5, 6], 'birthday-ish 3-7-10-13-17-25': [3, 7, 10, 13, 17, 25],
      'last Virada 9-13-21-32-33-59': [9, 13, 21, 32, 33, 59], 'spread high 34-40-46-50-52-58': [34, 40, 46, 50, 52, 58]}
out['examples_M'] = {k: round(M(v), 3) for k, v in ex.items()}
json.dump(out, open('c_results.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
