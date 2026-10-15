"""Parameter calibration: fit SMAP parameters, the Kt shape and rain-gauge weights to a flow history.

Replaces calibracao, calibracao_paralela, calibracao_pmur_paralela and their objective functions.
Parameters are named, and any parameter without bounds stays fixed, so a subset can be calibrated.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from . import weighting
from .assimilation import r_lbfgsb
from .kernel import simulate
from .metrics import OBJECTIVES
from .params import SmapParams, initial_state

SMAP_FIELDS = ["str_", "k2t", "crec", "capc", "k_kt", "h1", "k2t2", "ai", "lam", "pmur", "h", "k1t", "k3t",
               "pcof", "ecof", "ecof2"]


def make_kt(kt_max: int, kt_min: int, alfa: float, ex: float) -> np.ndarray:
    """63 Kt weights from a Beta(alfa, beta) shape over the kt_max+kt_min+1 day window (R's cria_kt)."""
    n = kt_max + kt_min + 1
    ex = 1 - ex
    b = alfa * (1 - ex) / ex
    start = 1 / (n + 2)
    q = start + np.arange(int(np.floor((1 - start) * n + 1e-10)) + 1) / n  # R's seq(start, 1, 1/n)
    aux = beta_dist.pdf(q, alfa, b)
    kt = np.zeros(63)
    kt[2 - kt_max: 3 + kt_min] = aux / aux.sum()
    return kt


@dataclass
class CalibrationResult:
    params: SmapParams
    gauges: pd.DataFrame  # nome, posto, valor (normalized weights)
    kt_shape: tuple[float, float] | None
    objective: float
    initial_objective: float  # with the starting parameters
    simulation: pd.Series  # daily Qcalc over the simulation period, with the fitted parameters
    weights: pd.Series  # objective weight per scored day
    log: pd.DataFrame  # one row per optimizer iteration: iteracao, the free parameters, objetivo
    optimizer: object  # scipy OptimizeResult; None when no parameter was free


def calibrate(p: SmapParams, rain: pd.DataFrame, gauges: pd.DataFrame, etp: pd.Series, flow: pd.Series,
              start, end, ebin: float, tuin: float, supin: float, bounds: dict[str, tuple[float, float]], *,
              objective_period=None, kt_shape: tuple[float, float] | None = None, pmur: bool = False,
              objective: int = 0, weights=None, eps: float = 1e-3, scale_by_upper: bool = False,
              maxiter: int = 100, workers: int | None = None) -> CalibrationResult:
    """Calibrate on the simulation period [start, end].

    rain: gauge rain (data, posto, valor) covering [start - kt_min, end + kt_max] days.
    etp: raw daily ETP indexed by date over [start, end]. flow: observed flow indexed by date.
    bounds: {name: (lower, upper)} for the free parameters; names are SmapParams fields, "alfa"/"ex"
    (Kt shape, needs `kt_shape` as the starting point) or gauge ids from `gauges.posto`.
    objective_period: (first, last) day scored, default the whole period; weights: per scored day.
    maxiter defaults to 100, R optim's default for L-BFGS-B.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(start, end)
    kt_max, kt_min = p.kt_max, p.kt_min
    gauge_ids = list(gauges["posto"])
    rain_days = pd.date_range(start - pd.Timedelta(days=kt_min), end + pd.Timedelta(days=kt_max))
    rain_wide = rain.pivot(index="data", columns="posto", values="valor").reindex(index=rain_days, columns=gauge_ids)
    if rain_wide.isna().any().any():
        raise ValueError("rain does not cover the simulation period plus the Kt window")
    rain_mat = rain_wide.to_numpy()
    etp_v = etp.reindex(days).to_numpy()
    if np.isnan(etp_v).any():
        raise ValueError("etp does not cover the simulation period")
    first, last = (pd.Timestamp(x) for x in (objective_period or (start, end)))
    scored = (days >= first) & (days <= last)
    q = flow.reindex(days[scored]).to_numpy()
    if np.isnan(q).any():
        raise ValueError("flow does not cover the objective period")
    q = np.where(q == 0, 0.1, q)  # as assimilation does: MAPE/DM divide by observed flow
    w =np.full(len(q), 1 / len(q)) if weights is None else np.asarray(weights, dtype=float)
    metric, sign = OBJECTIVES[objective]

    shape = dict(zip(("alfa", "ex"), kt_shape)) if kt_shape else {}
    current = {**{f: getattr(p, f) for f in SMAP_FIELDS}, **shape, **dict(zip(gauge_ids, gauges["valor"]))}
    unknown = set(bounds) - {k for k, v in current.items() if v is not None}
    if unknown:
        raise ValueError(f"cannot calibrate {sorted(unknown)}: unknown name or no starting value")
    free = list(bounds)
    lower = np.array([bounds[k][0] for k in free], dtype=float)
    upper = np.array([bounds[k][1] for k in free], dtype=float)
    upper[upper == lower] += 1e-6

    def unpack(x):
        v = {**current, **dict(zip(free, x))}
        kt = make_kt(kt_max, kt_min, v["alfa"], v["ex"]) if kt_shape else p.kt
        g = np.array([v[i] for i in gauge_ids])
        return dataclasses.replace(p, kt=kt, **{f: v[f] for f in SMAP_FIELDS}), g / g.sum()

    def run(x):
        pp, g = unpack(x)
        prec = weighting.temporal(rain_mat @ g * pp.pcof, pp.kt, kt_max, kt_min)
        return simulate(pp, initial_state(pp, ebin, tuin, supin), prec, etp_v * pp.ecof, etp_v * pp.ecof2, pmur)

    x0 = np.array([current[k] for k in free], dtype=float)
    f = lambda x: sign * metric(run(x)[scored, 0], q, w)  # noqa: E731
    log = []
    f0 = f(x0)
    if free:
        x, fx, res = r_lbfgsb(f, x0, lower, upper, eps, upper.copy() if scale_by_upper else None,
                              maxiter=maxiter, workers=workers,
                              callback=lambda x, fx: log.append([len(log) + 1, *x, sign * fx]))
    else:  # nothing to fit: score the current parameters
        x, fx, res = x0, f0, None
    fitted, g = unpack(x)
    v = {**current, **dict(zip(free, x))}
    return CalibrationResult(params=fitted, gauges=gauges.assign(valor=g),
                             kt_shape=(v["alfa"], v["ex"]) if kt_shape else None,
                             objective=sign * fx, initial_objective=sign * f0,
                             simulation=pd.Series(run(x)[:, 0], index=days, name="simulado"),
                             weights=pd.Series(w, index=days[scored], name="peso"),
                             log=pd.DataFrame(log, columns=["iteracao", *free, "objetivo"]), optimizer=res)
