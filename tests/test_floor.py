"""Exactness and gradient checks for src/floor_model.py."""
import itertools
import numpy as np
from math import comb
from scipy.optimize import approx_fprime
import floor_model as fm

rng = np.random.default_rng(0)


def random_draws(n):
    M = np.zeros((n, 60))
    for i in range(n):
        M[i, rng.choice(60, 6, replace=False)] = 1
    return M


def test_esp_matches_brute_force():
    w = rng.uniform(0.5, 2.0, 12)
    mask = (rng.random((5, 12)) < 0.6).astype(float)
    E = fm.esp(w, mask, order=6)
    for r in range(5):
        sel = w[mask[r] == 1]
        for j in range(7):
            brute = sum(np.prod(c) for c in itertools.combinations(sel, j)) if j else 1.0
            assert np.isclose(E[r, j], brute, rtol=1e-12)


def test_esp_minus_matches_recompute():
    w = rng.uniform(0.5, 2.0, 60); M = random_draws(4)
    E = fm.esp(w, M); R = fm.esp_minus(E, w, M)
    for r in range(4):
        for i in range(60):
            m2 = M[r].copy(); m2[i] = 0
            assert np.allclose(R[r, i], fm.esp(w, m2[None, :])[0], rtol=1e-10)


def test_P_values():
    assert np.isclose(1 / fm.P[4], 2332.0, rtol=1e-3) and np.isclose(1 / fm.P[5], 154518.0, rtol=1e-5)
    assert np.isclose(fm.P[6], 1 / comb(60, 6))


def test_uniform_weights_give_P():
    M = random_draws(20)
    for k in (4, 5, 6):
        assert np.allclose(fm.struct_Q(np.zeros(60), M, k), fm.P[k], rtol=1e-12)


def test_Q_sums_to_one_over_k():
    # every ticket matches some k of the drawn six: sum_k Q_k(d) = 1 exactly, for any weights
    t = rng.normal(0, 0.3, 60); t -= t.mean(); M = random_draws(30)
    tot = sum(fm.struct_Q(t, M, k) for k in range(1, 7)) + fm.esp(np.exp(t), 1 - M)[:, 6] / fm.esp(np.exp(t), np.ones((1, 60)))[0, 6]
    assert np.allclose(tot, 1.0, rtol=1e-12)


def test_Q_averages_to_P_over_uniform_draws():
    t = rng.normal(0, 0.3, 60); t -= t.mean(); M = random_draws(100_000)
    for k in (4, 5):
        q = fm.struct_Q(t, M, k) / fm.P[k]
        assert abs(q.mean() - 1) < 4 * q.std() / np.sqrt(len(q))


def test_Q6_is_product_of_weights():
    t = rng.normal(0, 0.3, 60); t -= t.mean(); M = random_draws(10); w = np.exp(t)
    e6 = fm.esp(w, np.ones((1, 60)))[0, 6]
    assert np.allclose(fm.struct_Q(t, M, 6), np.exp(M @ t) / e6, rtol=1e-12)


def _toy_data(n=300):
    M = random_draws(n); s = rng.uniform(1e6, 5e6, n); g = rng.integers(0, 5, n)
    t = rng.normal(0, 0.2, 60); t -= t.mean()
    tiers = {k: rng.poisson(s * 1e-3 * (0.3 * fm.P[k] + 0.7 * fm.struct_Q(t, M, k)) / fm.P[k] * fm.P[k] * 50).astype(float) + 1
             for k in (4, 5)}
    return fm.Data(M, s, g, tiers, rng.random(n) < 0.05)


def test_struct_gradient():
    data = _toy_data(); phi = {4: 3.0, 5: 1.5}
    x = np.r_[rng.normal(0, 0.1, 59), 0.2, -0.1, 0.3]   # theta, v4, v5, f
    f = lambda z: fm.struct_obj(z, data, phi)[0]
    num = approx_fprime(x, f, 1e-6)
    assert np.allclose(fm.struct_obj(x, data, phi)[1], num, rtol=1e-4, atol=1e-5)
    g = lambda z: fm.struct_obj(z, data, phi, f_fixed=0.2)[0]
    assert np.allclose(fm.struct_obj(x[:61], data, phi, f_fixed=0.2)[1], approx_fprime(x[:61], g, 1e-6), rtol=1e-4, atol=1e-5)


def test_loglin_gradient():
    data = _toy_data(); x = np.r_[rng.normal(0, 0.1, 59), 0.15]
    f = lambda z: fm.loglin_obj(z, data, 4)[0]
    assert np.allclose(fm.loglin_obj(x, data, 4)[1], approx_fprime(x, f, 1e-6), rtol=1e-4, atol=1e-5)


def test_recovers_f_without_overdispersion():
    # big counts, pure Poisson, strong popularity: f must come back close to truth
    n = 2000; M = random_draws(n); s = np.full(n, 1e8); g = np.arange(n) % 8
    t = rng.normal(0, 0.35, 60); t -= t.mean(); f_true = 0.3
    tiers = {k: rng.poisson(s * (f_true * fm.P[k] + (1 - f_true) * fm.struct_Q(t, M, k))).astype(float) for k in (4, 5)}
    r = fm.fit_struct(fm.Data(M, s, g, tiers, (np.arange(n) % 97 == 0)), grid=np.arange(0, 0.61, 0.1))
    assert abs(r['f'] - f_true) < 0.03 and np.corrcoef(r['theta'], t)[0, 1] > 0.99


def test_fast_Q_matches_direct_esp():
    t = rng.normal(0, 0.4, 60); t -= t.mean(); w = np.exp(t); M = random_draws(50)
    Ein, Eout, Eall = fm.esp(w, M), fm.esp(w, 1 - M), fm.esp(w, np.ones((1, 60)))[0]
    for k in (4, 5, 6):
        assert np.allclose(fm.struct_Q(t, M, k), Ein[:, k] * Eout[:, 6 - k] / Eall[6], rtol=1e-11)
