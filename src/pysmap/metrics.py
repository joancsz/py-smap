"""Weighted goodness-of-fit metrics. `w` defaults to uniform weights summing to 1."""
from __future__ import annotations

import numpy as np


def _w(obs, w):
    return np.full(len(obs), 1 / len(obs)) if w is None else np.asarray(w, dtype=float)


def nse(sim, obs, w=None):
    w = _w(obs, w)
    return 1 - np.sum((obs - sim) ** 2 * w) / np.sum((obs - np.average(obs, weights=w)) ** 2 * w)


def mape(sim, obs, w=None):
    return np.sum(np.abs((obs - sim) / obs) * _w(obs, w))


def dm(sim, obs, w=None):
    """Distance to the ideal point (MAPE=0, NSE=1): the ONS default assimilation objective."""
    return np.sqrt(mape(sim, obs, w) ** 2 + (1 - nse(sim, obs, w)) ** 2)


def pbias(sim, obs, w=None):
    w = _w(obs, w)
    return np.sum(sim * w) / np.sum(obs * w)


def correlation(sim, obs, w=None):
    w = _w(obs, w)
    do, ds = obs - np.sum(obs * w), sim - np.sum(sim * w)
    return np.sum(w * do * ds) / (np.sqrt(np.sum(w * do ** 2)) * np.sqrt(np.sum(w * ds ** 2)))


def alpha(sim, obs, w=None):
    w = _w(obs, w)
    return np.sqrt(np.sum(w * (sim - np.sum(sim * w)) ** 2)) / np.sqrt(np.sum(w * (obs - np.sum(obs * w)) ** 2))


def kge(sim, obs, w=None):
    return 1 - np.sqrt((correlation(sim, obs, w) - 1) ** 2 + (pbias(sim, obs, w) - 1) ** 2
                       + (alpha(sim, obs, w) - 1) ** 2)


def rmse(sim, obs, w=None):
    return np.sqrt(np.sum((sim - obs) ** 2 * _w(obs, w)))


# `funcao_objetivo` code in inicializacao.csv -> (metric, sign). Optimizers minimize sign * metric.
OBJECTIVES = {0: (dm, 1), 1: (nse, -1), 2: (mape, 1), 3: (kge, -1)}
