"""Flow routing and totalization: from sub-basin incremental forecasts to total flow at each station."""
from __future__ import annotations

import warnings
from functools import lru_cache
from graphlib import TopologicalSorter

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def propagate_tv(upstream, downstream, travel_hours):
    """Shift upstream flow by a travel time (hours), split over two days, and add it downstream.

    The first ceil(tv/24) days stay 0, as in smapOnsR.
    """
    n = len(downstream)
    out = np.zeros(n)
    lag = int(np.ceil(travel_hours / 24.0))
    f2 = (24 * lag - travel_hours) / 24.0
    f1 = (travel_hours - 24 * (lag - 1)) / 24.0
    for i in range(lag, n):
        if lag > 0:
            out[i] = upstream[i - lag + 1] * f2 + upstream[i - lag] * f1 + downstream[i]
        else:
            out[i] = upstream[i] * f2 + upstream[i] * f1 + downstream[i]
    return out


@njit(cache=True)
def propagate_muskingum(upstream, downstream, n_reaches, c1, c2, c3):
    """Route upstream flow through n Muskingum reaches and add it downstream. Day 0 stays 0, as in smapOnsR."""
    n = len(downstream)
    q = np.empty((n, n_reaches + 1))
    q[0, :] = upstream[0]
    q[:, 0] = upstream
    out = np.zeros(n)
    for i in range(1, n):
        for j in range(1, n_reaches + 1):
            q[i, j] = q[i, j - 1] * c1 + q[i - 1, j - 1] * c2 + q[i - 1, j] * c3
        out[i] = q[i, n_reaches] + downstream[i]
    return out


def muskingum_coeffs(k_days: float, x: float, nf: int) -> tuple[float, float, float]:
    """(c1, c2, c3) for propagate_muskingum from MPV's K (days), x and phases, daily step.

    MPV ExecutorMuskingum.CalcularValoresIniciaisPreCalculoMK; MPV also rounds each phase output to
    4 decimals, which is not reproduced here (< 1e-3 m3/s vs MPV in tests/data/mpv_golden.csv).
    """
    if x >= 0.5:  # MPV only warns too
        warnings.warn(f"Muskingum x = {x} >= 0.5 gives negative coefficients", stacklevel=2)
    kt = k_days / nf
    den = kt - kt * x + 0.5
    return (0.5 - kt * x) / den, (0.5 + kt * x) / den, (kt - kt * x - 0.5) / den


@njit(cache=True)
def _ssarr_phase(i1, i2, o1, nf, table_q, table_t):
    """One SSARR phase over one day: Newton solve for the end-of-day outflow (MPV ExecutorSSARR.Propagar1Fase).

    Storage time Ts (h) comes from the reach's Q x T table, interpolated at the mean outflow and split over nf
    phases. Iteration limits (100, 0.11 m3/s) are hard-coded in MPV; the file's values are never used.
    """
    im = (i1 + i2) / 2.0
    o2 = o1
    last = len(table_q) - 1
    j = 0  # MPV keeps the row from the previous iteration when qbar falls below the table
    for _ in range(100):
        qbar = (o1 + o2) / 2.0
        for k in range(last + 1):
            if table_q[k] <= qbar:
                j = k
        if j == last:
            j -= 1
        qa, qb, ta, tb = table_q[j], table_q[j + 1], table_t[j], table_t[j + 1]
        if qb == qa:
            qb = qa + 0.001
        if qb == qbar:
            qb = qbar + 0.001
        slope = (tb - ta) / (qb - qa)
        ts = (tb + slope * (qbar - qb)) / nf
        f = (o2 - o1) - (im - o1) * 24.0 / (ts + 12.0)
        # MPV divides by (Ts+12)^2; the exact derivative would have 2*(Ts+12)^2.
        fp = 1.0 + 24.0 * (im - o1) * (slope / nf) / (ts + 12.0) ** 2
        new = o2 - f / fp
        err = abs(o2 - new)
        o2 = new
        if err <= 0.11:
            break
    return o2


@njit(cache=True)
def propagate_ssarr(upstream, downstream, nf, table_q, table_t):
    """Route upstream flow through nf SSARR phases and add it downstream (MPV ExecutorSSARR.ExecutarModeloAsync).

    Day 0 sets every phase to the inflow. Phase k's inflow is phase k-1's outflow; the reach output is
    truncated to an integer, as MPV does (inherited from CPINS), but intermediate phases are not.
    """
    n = len(upstream)
    state = np.full(nf, upstream[0])
    out = np.empty(n)
    out[0] = np.trunc(upstream[0]) + downstream[0]
    for t in range(1, n):
        i_prev, i_now = upstream[t - 1], upstream[t]
        for f in range(nf):
            o = _ssarr_phase(i_prev, i_now, state[f], nf, table_q, table_t)
            i_prev, i_now = state[f], o
            state[f] = o
        out[t] = np.trunc(i_now) + downstream[t]
    return out


@njit(cache=True)
def _todini_table(b1, b2, n1, n2, s0, hi, dt_s, length_m):
    """Qref, Courant and diffusion numbers for depths 0.01..60 m, two-stage section (MPV PreecherTabelaPasso1).

    Co and D are blended by each stage's share of Qref, as MPV's CalcularVazaoSaida does; MPV's D there is
    Qref / (ce + beta + B*X*S0), a sum where the textbook has a product, kept as in MPV.
    """
    m = 6000
    qref, co, d = np.empty(m), np.empty(m), np.empty(m)
    h, r = 0.01, 0
    while r < m:
        hhi = max(h - hi, 0.0)
        q1 = h ** 1.66666666666666 * b1 * s0 ** 0.5 / n1
        q2 = hhi ** 1.66666666666666 * b2 * s0 ** 0.5 / n2
        ce1 = 1.66666666 * s0 ** 0.5 * h ** 0.6666666666 * (1.0 - 4.0 * h / (5.0 * b1)) / n1
        ce2 = 1.66666666 * s0 ** 0.5 * hhi ** 0.6666666666 * (1.0 - 4.0 * hhi / (5.0 * b2)) / n2
        beta1 = ce1 * b1 * h / q1
        beta2 = 0.0 if q2 == 0 else ce2 * hhi * b2 / q2
        co2 = 0.0 if q2 == 0 else ce2 * dt_s / (beta2 * length_m)
        qr = q1 + q2
        w1, w2 = q1 / qr, q2 / qr
        qref[r] = qr
        co[r] = ce1 * dt_s / (beta1 * length_m) * w1 + co2 * w2
        d[r] = qr / (ce1 + beta1 + b1 * length_m * s0) * w1 + qr / (ce2 + beta2 + b2 * length_m * s0) * w2
        h += 0.01
        r += 1
        if not h < 60.0:
            break
    return qref[:r], co[:r], d[:r]


@njit(cache=True)
def _todini_row(qref, q):
    """Row with Qref nearest to q (MPV linhaQref); clamped at the table ends, where MPV would fail."""
    k = np.searchsorted(qref, q, side="right")
    if k == 0:
        return 0
    if k == len(qref) or abs(q - qref[k - 1]) <= abs(q - qref[k]):
        return k - 1
    return k


@njit(cache=True)
def _todini_out(qref, co, d, o_t, i_t, i_t1, o_t1):
    r0 = _todini_row(qref, (i_t + o_t) / 2.0)
    r1 = _todini_row(qref, (i_t1 + o_t1) / 2.0)
    den = 1.0 + co[r1] + d[r1]
    ratio = co[r1] / co[r0]
    c1 = (-1.0 + co[r0] + d[r0]) / den
    c2 = (1.0 + co[r0] - d[r0]) / den * ratio
    c3 = (1.0 - co[r0] + d[r0]) / den * ratio
    return c1, c2, c3


@njit(cache=True)
def propagate_todini(upstream, downstream, b1, b2, n1, n2, s0, hi, length_m, dt_s=86400.0):
    """Variable-parameter Muskingum-Cunge (Todini 2007) and add it downstream (MPV ExecutorTodini).

    b1/b2 main/overbank widths (m), n1/n2 Manning, s0 slope, hi bankfull depth (m), length_m reach length.
    Each day runs twice (MPV CalcularVazaoSaida): first with O(t+1) guessed as O(t) + I(t+1) - I(t) to
    find the end-of-day reference flow, then with that pass's result. As in MPV, the value reported on day
    t is O(t+1), computed from I(t+1) (I(t) on the last day), and routing starts from O = I(0).
    """
    qref, co, d = _todini_table(b1, b2, n1, n2, s0, hi, dt_s, length_m)
    n = len(upstream)
    out = np.empty(n)
    o = upstream[0]
    for t in range(n):
        i_t, i_t1 = upstream[t], upstream[min(t + 1, n - 1)]
        guess = o + i_t1 - i_t
        c1, c2, c3 = _todini_out(qref, co, d, o, i_t, i_t1, guess)
        first = c1 * i_t1 + c2 * i_t + c3 * guess  # MPV's first pass weights the guess, not O(t)
        c1, c2, c3 = _todini_out(qref, co, d, o, i_t, i_t1, first)
        o = c1 * i_t1 + c2 * i_t + c3 * o
        out[t] = o + downstream[t]
    return out


@lru_cache
def _ssarr_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """SSARR Q (m3/s) x T (h) table: first two columns of a ';' CSV, sorted by flow."""
    t = pd.read_csv(path, sep=";", encoding="utf-8-sig").iloc[:, :2].to_numpy(float)
    if len(t) < 2 or (np.diff(t[:, 0]) < 0).any():
        raise ValueError(f"{path}: SSARR table needs >= 2 rows sorted by flow")
    return t[:, 0].copy(), t[:, 1].copy()


def _route(up: np.ndarray, down: np.ndarray, cfg) -> np.ndarray:
    """Route every scenario (row) of `up` into `down` with the reach's `metodo`.

    Blank metodo keeps smapOnsR's rule: travel time if n == 0, else Muskingum with c1..c3.
    """
    m = cfg.get("metodo", "")
    if m == "muskingum":
        c = muskingum_coeffs(cfg.k, cfg.x, int(cfg.nf))
        f = lambda u, d: propagate_muskingum(u, d, int(cfg.nf), *c)  # noqa: E731
    elif m == "ssarr":
        tq, tt = _ssarr_table(cfg.tabela)
        f = lambda u, d: propagate_ssarr(u, d, int(cfg.nf), tq, tt)  # noqa: E731
    elif m == "todini":
        f = lambda u, d: propagate_todini(u, d, cfg.b1, cfg.b2, cfg.n1, cfg.n2, cfg.s0, cfg.hi,  # noqa: E731
                                          cfg.comprimento)
    elif m == "sd" or (m == "" and cfg.n == 0):
        f = lambda u, d: propagate_tv(u, d, float(cfg.tv))  # noqa: E731
    elif m == "":
        f = lambda u, d: propagate_muskingum(u, d, int(cfg.n), cfg.c1, cfg.c2, cfg.c3)  # noqa: E731
    else:
        raise ValueError(f"posto {cfg.name}: unknown metodo {m!r} (sd, muskingum, ssarr, todini)")
    return np.vstack([f(u, d) for u, d in zip(up, down)])


def _order(cfg: pd.DataFrame) -> list:
    """Stations upstream-first; a regressed station also waits for its base station."""
    ts = TopologicalSorter({p: set() for p in cfg.index})
    for p, row in cfg.iterrows():
        if row.posto_jusante in cfg.index:
            ts.add(row.posto_jusante, p)
        if row.regressao > 0 and row.flag_regressao == 0 and row.posto_regredido in cfg.index:
            ts.add(p, row.posto_regredido)
    return list(ts.static_order())


def totalize(forecast: pd.DataFrame, observed: pd.DataFrame, config: pd.DataFrame, history_days: int = 90):
    """Port of smapOnsR's totaliza_previsao.

    forecast: Qcalc rows of `run_chained(...)["previsao"]` (data_caso, data_previsao, cenario, nome, valor).
    observed: sub-basin flow history (data, nome, valor); it warms up the routing before each run date.
    config: ONS configuracao.csv (posto, nome_real, sub_bacia_agrupada, bacia_smap, fator, regressao,
    posto_regredido, flag_regressao, posto_jusante, tv, n, c1, c2, c3). Optional MPV methods via `metodo`:
    sd (tv), muskingum (k days, x, nf), ssarr (nf, tabela = path to a vazao;tempo CSV),
    todini (b1, b2, n1, n2, s0, hi, comprimento in m).

    Returns data_rodada, cenario, data_previsao, nome (station), previsao_incremental, previsao_total,
    previsao_total_sem_tv (no routing delay) and previsao_inc_tv (total minus upstream totals).
    """
    if "variavel" in forecast:
        forecast = forecast[forecast["variavel"] == "Qcalc"]
    cfg = config.copy()
    mpv = cfg.columns.intersection(["k", "x", "nf", "b1", "b2", "n1", "n2", "s0", "hi", "comprimento"])
    for c in ("regressao", "posto_regredido", "flag_regressao", "posto_jusante", "tv", "n", "c1", "c2", "c3", *mpv):
        cfg[c] = pd.to_numeric(cfg[c]).fillna(0)
    if "metodo" in cfg:
        cfg["metodo"] = cfg["metodo"].fillna("").astype(str).str.strip().str.lower()
    cfg["key"] = cfg["sub_bacia_agrupada"].str.strip().str.lower()
    cfg = cfg.set_index("posto")
    name_of = cfg["nome_real"]
    out = []
    for d, fc in forecast.groupby("data_caso"):
        d = pd.Timestamp(d)
        basins = fc["nome"].unique()
        # shortcut: observed days >= d are dropped; R appended them and could duplicate dates.
        obs = observed[(observed["data"] < d) & (observed["data"] >= d - pd.Timedelta(days=history_days))
                       & observed["nome"].isin(basins)].rename(columns={"data": "data_previsao"})
        hist = obs.merge(pd.DataFrame({"cenario": fc["cenario"].unique()}), how="cross")
        q = pd.concat([hist, fc])[["cenario", "data_previsao", "nome", "valor"]]
        wide = q.pivot(index=["cenario", "data_previsao"], columns="nome", values="valor")
        wide.columns = wide.columns.str.strip().str.lower()
        scen = wide.index.get_level_values(0).unique()
        days = wide.index.get_level_values(1).unique().sort_values()
        wide = wide.reindex(pd.MultiIndex.from_product([scen, days]))  # full grid; gaps become NaN
        n_scen = len(scen)
        grid = lambda s: s.to_numpy().reshape(n_scen, -1)  # noqa: E731  (scenario, day)

        mine = cfg[cfg["key"].isin(wide.columns)]
        inc = {p: grid(wide[row.key]) * row.fator for p, row in mine.iterrows()}
        for p, row in mine[(mine.regressao > 0) & (mine.flag_regressao == 1)].iterrows():
            if row.posto_regredido in inc:
                inc[p] = inc[row.posto_regredido] * row.regressao

        flu = mine[mine["bacia_smap"] == "posto_flu"]
        for p in _order(flu):  # flow gauges route into the incremental of their downstream station
            j = flu.loc[p, "posto_jusante"]
            if j in inc:
                inc[j] = _route(inc[p], inc[j], flu.loc[p])

        sub = mine[mine["bacia_smap"] != "posto_flu"]
        order = _order(sub)
        total = {p: inc[p].copy() for p in sub.index}
        no_tv = {p: inc[p].copy() for p in sub.index}
        for p in order:
            row = sub.loc[p]
            base = row.posto_regredido
            if row.regressao > 0 and row.flag_regressao == 0 and base in total:
                inc[p] = total[base] * row.regressao
                total[p] = total[p] + total[base] * row.regressao
                no_tv[p] = no_tv[p] + no_tv[base] * row.regressao
            j = row.posto_jusante
            if j in total:
                total[j] = _route(total[p], total[j], row)
                no_tv[j] = no_tv[p] + no_tv[j]
        inc_tv = {p: total[p].copy() for p in sub.index}
        for p in order:
            j = sub.loc[p, "posto_jusante"]
            if j in inc_tv:
                inc_tv[j] = inc_tv[j] - total[p]

        keep = days >= d
        for p in sub.index:
            out.append(pd.DataFrame({
                "data_rodada": d, "cenario": np.repeat(scen, keep.sum()),
                "data_previsao": np.tile(days[keep], n_scen), "nome": name_of[p],
                **{k: v[p][:, keep].ravel() for k, v in (("previsao_incremental", inc), ("previsao_total", total),
                                                        ("previsao_total_sem_tv", no_tv),
                                                        ("previsao_inc_tv", inc_tv))}}))
    return pd.concat(out, ignore_index=True)
