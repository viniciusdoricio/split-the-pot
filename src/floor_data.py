"""Real-data inputs for the floor model: draw matrix, sales, quarter groups, Virada flag."""
import numpy as np
import pandas as pd
from common import load_draws, popularity_frame
import floor_model as fm


def load(tiers=(4, 5)):
    d = popularity_frame(load_draws())
    M = np.zeros((len(d), 60))
    for i, r in enumerate(d.d): M[i, np.asarray(r) - 1] = 1
    g = pd.factorize(d.period)[0]
    data = fm.Data(M, d.sales.values, g, {k: d[f'w{k}'].values.astype(float) for k in tiers}, d.virada.values)
    return d, data
