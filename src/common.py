"""Shared helpers: the draw frame used by the popularity models, and the Virada EV formula."""
import os
import numpy as np
import pandas as pd
from math import comb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'out')
C60 = comb(60, 6)


def load_draws():
    df = pd.read_pickle(os.path.join(OUT, 'draws.pkl'))
    df['dt'] = pd.to_datetime(df.date, dayfirst=True)
    return df


def popularity_frame(df):
    """Draws with sales and quadra data (the 2,020 used by every popularity model), with Virada flag and quarter label.
    Virada = drawn on 31 Dec (or 1 Jan 2026), or sales > 8x the rolling 9-draw median (catches specials such as contest 3010)."""
    d = df[(df.sales > 0) & (df.w4 > 0)].copy().reset_index(drop=True)
    d['virada'] = ((d.dt.dt.month == 12) & (d.dt.dt.day == 31)) | ((d.dt.dt.month == 1) & (d.dt.dt.day == 1) & (d.dt.dt.year == 2026))
    d['virada'] = d['virada'] | (d.sales > 8 * d.sales.rolling(9, center=True, min_periods=1).median())
    d['period'] = d.dt.dt.to_period('Q').astype(str)
    return d


def ev(Mc, B, sena_pool=1.0e9, M5=None, M4=None):
    """Expected R$ back per R$6 ticket at the Virada.
    Mc: jackpot popularity multiplier of the ticket (1 = average ticket; already includes any random-bet floor).
    B: number of 6-number combinations bought. M5/M4: quina/quadra multipliers; default Mc^(5/6), Mc^(4/6)."""
    M5 = Mc ** (5 / 6) if M5 is None else M5
    M4 = Mc ** (4 / 6) if M4 is None else M4
    mu = B / C60 * Mc
    share = (1 - np.exp(-mu)) / mu                       # E[1/(1+K)], K~Poisson(mu)
    jack = sena_pool / C60 * share
    total_rateio = sena_pool / 0.90
    q5 = 0.05 * total_rateio; q4 = 0.05 * total_rateio
    # quina/quadra: thousands of winners, share ~ pool / expected winners
    e5 = q5 / max(B / 154_518 * M5, 1) / 154_518
    e4 = q4 / max(B / 2_332 * M4, 1) / 2_332
    return {'mu_cowinners': round(mu, 2), 'expected_share_of_jackpot': round(share, 4),
            'jackpot_if_you_win_expected_R$': round(sena_pool * share), 'ev_jackpot': round(jack, 2),
            'ev_quina': round(e5, 3), 'ev_quadra': round(e4, 3), 'ev_total_per_R$6': round(jack + e5 + e4, 2),
            'return_pct': round((jack + e5 + e4) / 6 * 100, 1)}
