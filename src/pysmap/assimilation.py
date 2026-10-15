"""Data assimilation: fit daily rain (and optionally ETP) multipliers plus initial Ebin/Supin to observed flow.

One function replaces smapOnsR's assimilacao_{oficial,evapotranspiracao,pmur,pmur_etp} and their
objective functions; they differed only in which multipliers are free and in optimizer settings.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import weighting
from .kernel import simulate
from .metrics import OBJECTIVES
from .params import SmapParams, initial_state


@dataclass
class AssimilationResult:
    prec_weights: np.ndarray  # one multiplier per assimilation day, oldest first
    etp_weights: np.ndarray | None
    ebin: float
    supin: float
    objective: float
    simulation: np.ndarray  # (n_days, len(kernel.COLUMNS))
    table: pd.DataFrame  # R's `otimizacao`: otimizacao, limite_inferior, limite_superior, variavel


def objective_weights(n: int) -> np.ndarray:
    """Log-decaying weights that favour the most recent days; they sum to 1."""
    i = np.arange(n, 0, -1)
    return (np.log(i + 1) - np.log(i)) / np.log(n + 1)


def r_gradient(f, lb, ub, h, pool=None):
    """Central differences clipped to the bounds, as R's optim does for L-BFGS-B.

    With a thread `pool` the coordinates are evaluated in parallel (the Numba kernel releases the GIL).
    """
    def partial(z, i):
        hi, lo = min(z[i] + h, ub[i]), max(z[i] - h, lb[i])
        zz = z.copy()
        zz[i] = hi
        f1 = f(zz)
        zz[i] = lo
        return (f1 - f(zz)) / (hi - lo)

    def grad(z):
        if pool is None:
            return np.array([partial(z, i) for i in range(len(z))])
        return np.array(list(pool.map(lambda i: partial(z, i), range(len(z)))))
    return grad


def r_lbfgsb(f, x0, lower, upper, eps, scale=None, maxiter=1000, workers=None, callback=None):
    """Minimize f with L-BFGS-B using R optim's defaults (lmm=5, factr=1e7, pgtol=0) and parscale.

    Returns (x, f(x), scipy result), x in unscaled units. `callback(x, fx)` runs after each iteration.
    """
    scale = np.ones_like(x0) if scale is None else scale
    lb, ub = lower / scale, upper / scale
    fz = lambda z: f(z * scale)  # noqa: E731
    pool = ThreadPoolExecutor(workers) if workers and workers > 1 else None
    try:
        cb = (lambda intermediate_result: callback(intermediate_result.x * scale, intermediate_result.fun))             if callback else None
        res = minimize(fz, np.clip(x0 / scale, lb, ub), jac=r_gradient(fz, lb, ub, eps, pool), method="L-BFGS-B",
                       bounds=list(zip(lb, ub)), callback=cb,
                       options={"maxiter": maxiter, "maxcor": 5, "ftol": 1e7 * np.finfo(float).eps, "gtol": 0})
    finally:
        if pool is not None:
            pool.shutdown()
    return res.x * scale, res.fun, res


def assimilate(p: SmapParams, tuin: float, ebin: float, supin: float, prec: np.ndarray, etp: np.ndarray,
               etp_plain: np.ndarray, q_obs: np.ndarray, *, pmur: bool = False, assimilate_etp: bool = False,
               prec_bounds=(0.5, 2.0), etp_bounds=(0.5, 2.0), ebin_bounds=(0.8, 1.2), supin_bounds=(0.0, 2.0),
               objective: int = 0, adjust_last_day: bool = False, fit: bool = True) -> AssimilationResult:
    """Assimilate n = len(q_obs) days ending the day before the forecast date.

    fit=False skips the optimizer: unit weights and the given Ebin/Supin (an open-loop run).

    prec: sub-basin rain (before Pcof) for the n days plus kt_min days before and kt_max days after.
    etp, etp_plain: ETP already multiplied by Ecof / Ecof2, n days.
    Bounds are multipliers: rain and ETP weights directly, Ebin relative to `ebin`, Supin relative to
    the maximum observed flow. The last day's rain weight is pinned to 1.
    """
    n = len(q_obs)
    prec_w = weighting.temporal(np.asarray(prec) * p.pcof, p.kt, p.kt_max, p.kt_min)
    if len(prec_w) != n:
        raise ValueError(f"{p.name}: rain window gives {len(prec_w)} days, flow has {n}")
    q = np.where(np.asarray(q_obs, dtype=float) == 0, 0.1, q_obs)
    if np.unique(q).size == 1:
        q[0] += 0.0001  # constant series make NSE undefined
    metric, sign = OBJECTIVES[objective]
    w_obj = objective_weights(n)
    n_etp = n if assimilate_etp else 0
    qmax = np.max(q_obs)

    lower = np.r_[np.full(n, prec_bounds[0]), np.full(n_etp, etp_bounds[0]),
                  ebin_bounds[0] * ebin, supin_bounds[0] * qmax]
    upper = np.r_[np.full(n, prec_bounds[1]), np.full(n_etp, etp_bounds[1]),
                  ebin_bounds[1] * ebin, supin_bounds[1] * qmax]
    lower[n - 1], upper[n - 1] = 0.9999999999, 1.0000000001
    # R parity: these settings differ between the four R variants for no documented reason.
    upper[upper == lower] += 1e-6 if assimilate_etp else 1e-8
    eps = 1e-4 if (pmur and assimilate_etp) else 1e-6
    scale = upper.copy() if pmur else None

    def run(x):
        ew = x[n:n + n_etp] if assimilate_etp else 1.0
        s0 = initial_state(p, x[-2], tuin, max(x[-1], 0.0))  # L-BFGS-B can return -1e-17
        return simulate(p, s0, prec_w * x[:n], etp * ew, etp_plain * ew, pmur)

    x0 = np.r_[np.ones(n + n_etp), ebin, supin]
    f = lambda x: sign * metric(run(x)[:, 0], q, w_obj)  # noqa: E731
    x, fx = r_lbfgsb(f, x0, lower, upper, eps, scale)[:2] if fit else (x0, f(x0))
    if adjust_last_day:
        x[n - 1] = min(x[n - 2], 1.0)

    names = [f"prec (t-{i})" for i in range(n, 0, -1)]
    names += [f"etp (t-{i})" for i in range(n, 0, -1)] if assimilate_etp else []
    table = pd.DataFrame({"otimizacao": x, "limite_inferior": lower, "limite_superior": upper,
                          "variavel": names + ["Ebin", "Supin"]})
    return AssimilationResult(prec_weights=x[:n], etp_weights=x[n:n + n_etp] if assimilate_etp else None,
                              ebin=x[-2], supin=max(x[-1], 0.0), objective=sign * fx, simulation=run(x),
                              table=table)
