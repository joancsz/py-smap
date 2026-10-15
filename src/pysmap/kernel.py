"""SMAP/ONS daily water balance (Numba). Port of funcaoSmapCpp's four kernels as one function.

Equation numbers refer to the ONS SMAP methodology manual, as in the original C++.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .params import SmapParams

# Ed2 = second surface outflow (Eq.16), Ed3 = flood-plain outflow (Eq.17): the C++ naming,
# which is what ONS output files carry. The pure-R kernel had these two names swapped.
COLUMNS = ("Qcalc", "Rsolo", "Rsup", "Rsup2", "Rsub", "Es", "Er", "Rec", "Marg", "Ed", "Ed3", "Ed2",
           "Eb", "Tu", "Qsup1", "Qplan", "Qsup2", "Qbase", "Ai")
COL = {c: i for i, c in enumerate(COLUMNS)}


@njit(cache=True, nogil=True)
def _simulate(str_, k2t, crec, capc, k_kt, h1, k2t2, ai, h, k1t, k3t, pmur_mode, lam, pmur, area,
              state0, prec, etp, emarg):
    n_scen, n_days = prec.shape
    out = np.empty((n_scen, n_days, 19))
    capc_tmp = capc / 100 * str_
    pmur_tmp = pmur / 100 * capc_tmp
    k_kts = 0.5 ** (1.0 / k_kt)
    k_1ts = 0.5 ** (1.0 / k1t)
    k_2ts = 0.5 ** (1.0 / k2t)
    k_2t2s = 0.5 ** (1.0 / k2t2)
    k_3ts = 0.5 ** (1.0 / k3t)
    for s in range(n_scen):
        rsolo, rsup, rsup2, rsub = state0[s, 0], state0[s, 1], state0[s, 2], state0[s, 3]
        for d in range(n_days):
            p = prec[s, d]
            e = etp[s, d]
            o = out[s, d]
            free = str_ - rsolo
            ai_d = free * lam / 100 if pmur_mode else ai
            tu = rsolo / str_  # Eq.19
            es = (p - ai_d) ** 2 / (p - ai_d + free) if p > ai_d else 0.0  # Eq.11
            if pmur_mode and rsolo <= pmur_tmp:
                er = 0.0
            elif p - es > e:
                er = e
            else:
                er = (p - es) + (e - (p - es)) * tu  # Eq.12
            rec = crec / 100 * tu * (rsolo - capc_tmp) if rsolo > capc_tmp else 0.0  # Eq.13
            marg = (rsup - h) * (1 - k_1ts) if rsup > h else 0.0  # Eq.14
            ed = min(rsup - marg, h1) * (1 - k_2ts)  # Eq.15
            ed3 = rsup2 * (1 - k_3ts)  # Eq.17
            ed2 = max(rsup - h1 - marg, 0.0) * (1 - k_2t2s)  # Eq.16
            eb = rsub * (1 - k_kts)  # Eq.18

            balance = rsolo + p - es - er - rec
            new_rsolo = min(balance, str_)
            new_rsub = rsub + rec - eb
            new_rsup = rsup + es - marg - ed - ed2 + max(0.0, balance - str_)
            new_rsup2 = max(rsup2 + marg - ed3 - emarg[s, d], 0.0)

            o[1], o[2], o[3], o[4] = new_rsolo, new_rsup, new_rsup2, new_rsub
            o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13] = es, er, rec, marg, ed, ed3, ed2, eb, tu
            o[14], o[15], o[16], o[17] = ed * area / 86.4, ed3 * area / 86.4, ed2 * area / 86.4, eb * area / 86.4
            o[0] = o[14] + o[15] + o[16] + o[17]
            o[18] = ai_d
            rsolo, rsup, rsup2, rsub = new_rsolo, new_rsup, new_rsup2, new_rsub
    return out


def simulate(p: SmapParams, state0, prec, etp, emarg, pmur: bool = False) -> np.ndarray:
    """Run SMAP for every scenario. Inputs are (n_days,) or (n_scen, n_days); state0 is (4,) or (n_scen, 4).

    Returns (n_scen, n_days, len(COLUMNS)); a 1-D input returns (n_days, len(COLUMNS)).
    """
    single = np.ndim(prec) == 1
    prec2 = np.atleast_2d(np.asarray(prec, dtype=float))
    shape = prec2.shape
    etp2 = np.broadcast_to(np.asarray(etp, dtype=float), shape)
    emarg2 = np.broadcast_to(np.asarray(emarg, dtype=float), shape)
    st = np.broadcast_to(np.asarray(state0, dtype=float), (shape[0], 4))
    if pmur and (p.lam is None or p.pmur is None):
        raise ValueError(f"sub-basin {p.name}: PMUR mode needs Lambda and Pmur")
    if not pmur and p.ai is None:
        raise ValueError(f"sub-basin {p.name}: classic mode needs Ai")
    out = _simulate(p.str_, p.k2t, p.crec, p.capc, p.k_kt, p.h1, p.k2t2, p.ai or 0.0, p.h, p.k1t, p.k3t,
                    pmur, p.lam or 0.0, p.pmur or 0.0, p.area,
                    np.ascontiguousarray(st), np.ascontiguousarray(prec2),
                    np.ascontiguousarray(etp2), np.ascontiguousarray(emarg2))
    return out[0] if single else out
