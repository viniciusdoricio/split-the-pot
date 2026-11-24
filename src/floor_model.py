"""Structural popularity model with a random-bet floor (ROADMAP item 1).

Bettors: a share f picks uniformly at random; the rest pick combination c with probability
    pi(c) ∝ prod_{i in c} w_i,   w_i = exp(theta_i),  sum(theta) = 0.
For draw d, expected tickets matching exactly k numbers (per unit of exposure):
    m_k(d) = f * P_k + (1 - f) * Q_k(d),   Q_k(d) = e_k(w_d) * e_{6-k}(w_notd) / e_6(w_all)
where e_j are elementary symmetric polynomials. Q_k averages to P_k over a uniform draw, so no
extra normalisation is needed. Winner counts: W_k,d ~ quasi-Poisson(sales_d * exp(alpha_{k,g(d)}) * m_k(d)),
with alpha per tier and per group g (quarter, Virada draws in their own group). Given the multiplier m,
alpha has a closed form, so it is profiled out and the optimiser only sees theta (59 free) and f.

The same machinery fits the current log-linear model (m = exp(sum of beta over the drawn numbers)),
so both are compared on identical groups and exposure.
"""
import numpy as np
from math import comb
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

P = {k: comb(6, k) * comb(54, 6 - k) / comb(60, 6) for k in range(7)}


# ---------- elementary symmetric polynomials ----------
def esp(w, mask, order=6):
    """e_0..e_order of the weights w (n_numbers,) selected by mask (n, n_numbers). Returns (n, order+1)."""
    E = np.zeros((mask.shape[0], order + 1)); E[:, 0] = 1.0
    for i in range(len(w)):
        x = (w[i] * mask[:, i])[:, None]
        E[:, 1:] = E[:, 1:] + x * E[:, :-1]          # RHS uses the old E: e_j += x * e_{j-1}
    return E


def esp_minus(E, w, mask):
    """e_j(S minus {i}) for every i: (n, n_numbers, order+1). For i not in S it equals e_j(S)."""
    n, J = E.shape; R = np.empty((n, len(w), J)); R[:, :, 0] = 1.0
    X = w[None, :] * mask
    for j in range(1, J):
        R[:, :, j] = E[:, [j]] - X * R[:, :, j - 1]
    return R


def theta60(t59):
    return np.r_[t59, -t59.sum()]


# ---------- multipliers ----------
def _remove(E, x):
    """e_j(S minus {x}) from e_j(S): R_j = E_j - x * R_{j-1}. E (..., J), x broadcastable to E[..., 0]."""
    R = np.empty_like(E); R[..., 0] = 1.0
    for j in range(1, E.shape[-1]):
        R[..., j] = E[..., j] - x * R[..., j - 1]
    return R


def struct_Qs(theta, M, ks, grad=False):
    """Q_k(d) for each k in ks (and dQ/dtheta, (n, 60), if grad). Uses only the 6 drawn weights per row:
    e(drawn) by a 6-step recursion, e(not drawn) by removing the 6 drawn weights from e(all)."""
    w = np.exp(theta); n = M.shape[0]
    idx = np.nonzero(M)[1].reshape(n, 6)
    Wd = w[idx]                                               # (n, 6)
    Ein = np.zeros((n, 7)); Ein[:, 0] = 1.0
    for c in range(6):
        Ein[:, 1:] = Ein[:, 1:] + Wd[:, [c]] * Ein[:, :-1]
    Eall = esp(w, np.ones((1, len(w))))[0]
    Eout = np.broadcast_to(Eall, (n, 7)).copy()
    for c in range(6):
        Eout = _remove(Eout, Wd[:, c])
    out = {}
    for k in ks:
        Q = Ein[:, k] * Eout[:, 6 - k] / Eall[6]
        if not grad:
            out[k] = Q; continue
        # d log Q / d w_i: drawn i -> e_{k-1}(d minus i)/e_k(d); not drawn i -> e_{5-k}(out minus i)/e_{6-k}(out); minus e_5(all minus i)/e_6(all)
        Rin = _remove(np.broadcast_to(Ein[:, None, :k], (n, 6, k)), Wd)[..., k - 1]                 # (n, 6)
        dlog = np.zeros((n, 60))
        if 6 - k >= 1:
            X = w[None, :] * (1.0 - M)
            Rout = _remove(np.broadcast_to(Eout[:, None, :6 - k], (n, 60, 6 - k)), X)[..., 5 - k]     # (n, 60)
            dlog = (1.0 - M) * Rout / Eout[:, [6 - k]]
        np.put_along_axis(dlog, idx, Rin / Ein[:, [k]], axis=1)
        Rall = _remove(np.broadcast_to(Eall[:6], (60, 6)), w)[:, 5]                                 # (60,)
        dlog = (dlog - (Rall / Eall[6])[None, :]) * w[None, :]
        out[k] = (Q, Q[:, None] * dlog)
    return out


def struct_Q(theta, M, k, grad=False):
    return struct_Qs(theta, M, (k,), grad)[k]


def struct_m(theta, f, M, k):
    """Floor-mixed multiplier relative to an average ticket: m_k / P_k (1 = uniform)."""
    return f + (1 - f) * struct_Q(theta, M, k) / P[k]


def loglin_m(beta, M, kappa=1.0):
    return np.exp(kappa * (M @ beta))


# ---------- profiled quasi-Poisson likelihood ----------
def profiled(y, s, g, m, ng):
    """Profiled Poisson log-lik (up to constants not involving m or the Virada offset in s) and d ell / d m."""
    S = np.bincount(g, s * m, ng); Y = np.bincount(g, y, ng)
    ok = Y > 0
    ll = np.sum(y * np.log(s * m)) - np.sum(Y[ok] * np.log(S[ok]))
    gm = y / m - (Y / np.where(S > 0, S, 1))[g] * s
    return ll, gm


def alphas(y, se, g, m, ng):
    """Closed-form group effects exp(alpha_g) = Y_g / sum_g(se * m), se = effective sales (Virada offset applied)."""
    return np.bincount(g, y, ng) / np.bincount(g, se * m, ng)


def pearson_phi(y, se, g, m, ng, n_par):
    mu = se * alphas(y, se, g, m, ng)[g] * m
    return float(np.sum((y - mu) ** 2 / mu) / (len(y) - ng - n_par))


class Data:
    """Draw matrix + per-tier counts. g: quarter codes; vir: Virada flag (additive tier-specific log offset,
    the same spec as b_popularity.py). tiers: dict k -> y (only the tiers used for fitting)."""
    def __init__(self, M, s, g, tiers, vir):
        self.M, self.s = M.astype(float), s.astype(float)
        self.g, self.ng = g, int(g.max()) + 1
        self.tiers, self.vir = tiers, vir.astype(float)

    def subset(self, idx):
        _, g = np.unique(self.g[idx], return_inverse=True)
        return Data(self.M[idx], self.s[idx], g, {k: y[idx] for k, y in self.tiers.items()}, self.vir[idx])

    def se(self, v):
        return self.s * np.exp(v * self.vir)


N_PAR = 60   # 59 free popularity parameters + Virada offset, per tier (f adds one when free; ignored in phi df)


def struct_obj(x, data, phi, f_fixed=None):
    """Negative weighted profiled log-lik and gradient. x = [theta(59), v_k per tier, f (if f_fixed is None)]."""
    K = list(data.tiers); nk = len(K)
    t = theta60(x[:59]); vs = x[59:59 + nk]; f = x[59 + nk] if f_fixed is None else f_fixed
    val, gt, gv, gf = 0.0, np.zeros(60), np.zeros(nk), 0.0
    QQ = struct_Qs(t, data.M, K, grad=True)
    for j, k in enumerate(K):
        y = data.tiers[k]
        Q, dQ = QQ[k]
        m = f * P[k] + (1 - f) * Q
        ll, gm = profiled(y, data.se(vs[j]), data.g, m, data.ng)
        val += ll / phi[k]
        gt += (1 - f) * (gm @ dQ) / phi[k]
        gv[j] = np.sum(data.vir * gm * m) / phi[k]
        gf += np.sum(gm * (P[k] - Q)) / phi[k]
    g = np.r_[gt[:59] - gt[59], gv] if f_fixed is not None else np.r_[gt[:59] - gt[59], gv, gf]
    sc = len(data.s)
    return -val / sc, -g / sc


def fit_fixed(data, f, phi, x0=None):
    """Fit theta and Virada offsets with f held fixed (well conditioned; f itself is profiled, never optimised jointly:
    jointly the f direction is nearly flat and L-BFGS stops early on it)."""
    K = list(data.tiers); nk = len(K)
    x0 = np.zeros(59 + nk) if x0 is None else np.asarray(x0, float)[:59 + nk]
    r = minimize(struct_obj, x0, args=(data, phi, f), jac=True, method='L-BFGS-B', bounds=[(-3.0, 3.0)] * (59 + nk),
                 options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-9})   # |theta| <= 3 only stops wild line-search steps
    return {'theta': theta60(r.x[:59]), 'v': dict(zip(K, map(float, r.x[59:]))), 'f': float(f),
            'loglik': float(-r.fun * len(data.s)), 'phi': phi, 'x': r.x.copy(), 'converged': bool(r.success)}


def estimate_phi(data, f=0.0, x0=None):
    """Pearson dispersion per tier from an unweighted fit at f."""
    r = fit_fixed(data, f, {k: 1.0 for k in data.tiers}, x0)
    t = r['theta']
    phi = {k: pearson_phi(y, data.se(r['v'][k]), data.g, f * P[k] + (1 - f) * struct_Q(t, data.M, k), data.ng, N_PAR)
           for k, y in data.tiers.items()}
    return phi, r


def fit_struct(data, f_fixed=None, phi=None, grid=None, x0=None, crit=3.841, xtol=0.004, ci=True):
    """f_fixed given: fit at that f. Otherwise: profile log-lik over a grid of f (warm-started), refine the maximum with
    a bounded 1-D search, and get the CI {f: 2*(ll_max - ll(f)) < crit} by bisection. phi (per tier) is estimated
    once at f = 0 (or f_fixed) unless given, and held fixed across the profile."""
    if phi is None:
        phi, r0 = estimate_phi(data, 0.0 if f_fixed is None else f_fixed, x0); x0 = r0['x']
    if f_fixed is not None:
        return fit_fixed(data, f_fixed, phi, x0)
    grid = np.round(np.arange(0, 0.901, 0.05), 4) if grid is None else np.asarray(grid)
    prof, xs, x = [], [], x0
    for f in grid:
        r = fit_fixed(data, f, phi, x); prof.append(r['loglik']); xs.append(r['x']); x = r['x']
    prof = np.array(prof); i = int(prof.argmax())
    def at(f):
        return fit_fixed(data, f, phi, xs[int(np.argmin(np.abs(grid - f)))])
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    cands = [fit_fixed(data, grid[i], phi, xs[i])]
    if hi_b > lo_b:
        rs = minimize_scalar(lambda f: -at(f)['loglik'], bounds=(lo_b, hi_b), method='bounded', options={'xatol': 1e-3})
        cands.append(at(float(rs.x)))
    best = max(cands, key=lambda r: r['loglik'])
    llmax = best['loglik']; D = 2 * (llmax - prof); fh = best['f']
    best.update({'profile_f': grid.tolist(), 'profile_D': D.tolist()})
    if not ci:
        return best
    Dfun = lambda f: 2 * (llmax - at(f)['loglik'])
    def root(a, b):                       # D(a) < crit <= D(b)
        while abs(b - a) > xtol:
            c = 0.5 * (a + b)
            if Dfun(c) < crit: a = c
            else: b = c
        return 0.5 * (a + b)
    left = [g for g, dd in zip(grid, D) if g < fh and dd >= crit]
    right = [g for g, dd in zip(grid, D) if g > fh and dd >= crit]
    f_lo = root(fh, max(left)) if left else float(grid[0])
    f_hi = root(fh, min(right)) if right else float(grid[-1])
    best.update({'ci': (float(f_lo), float(f_hi)),
                 'ci_open_low': not left, 'ci_open_high': not right, 'D_at_0': float(D[0]) if grid[0] == 0 else None})
    return best


# ---------- log-linear (current) model in the same framework ----------
def loglin_obj(x, data, k):
    b = theta60(x[:59]); v = x[59]; y = data.tiers[k]
    m = np.exp(data.M @ b)
    ll, gm = profiled(y, data.se(v), data.g, m, data.ng)
    gb = (gm * m) @ data.M
    return -ll / len(y), -np.r_[gb[:59] - gb[59], np.sum(data.vir * gm * m)] / len(y)


def fit_loglin(data, k=4):
    """Current model: log E[W_k] = log sales + quarter FE + v*virada + sum(beta). Returns (beta(60), v)."""
    r = minimize(loglin_obj, np.zeros(60), args=(data, k), jac=True, method='L-BFGS-B',
                 options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10})
    return theta60(r.x[:59]), float(r.x[59])


# ---------- jackpot (W6) scoring ----------
def pois_logpmf(y, mu):
    return y * np.log(mu) - mu - gammaln(y + 1)


def w4_exposure(data, y4, v4, m4):
    """log(sales) + Virada offset + alpha4_g, with alpha4 estimated from W4 with the popularity multiplier frozen."""
    se = data.se(v4)
    return np.log(se) + np.log(alphas(y4, se, data.g, m4, data.ng))[data.g]


def w6_calibrate(y6, offset, vir, extra=None):
    """Poisson GLM for jackpot winners with a fixed offset: log mu6 = offset + c + v*virada (+ kappa*extra)."""
    import statsmodels.api as sm
    X = np.column_stack([np.ones(len(y6)), vir] + ([extra] if extra is not None else []))
    return sm.GLM(y6, X, family=sm.families.Poisson(), offset=offset).fit().params


def w6_mu(params, offset, vir, extra=None):
    X = np.column_stack([np.ones(len(offset)), vir] + ([extra] if extra is not None else []))
    return np.exp(offset + X @ params)
