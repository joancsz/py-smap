"""Flow routing and totalization: from sub-basin incremental forecasts to total flow at each station."""
from __future__ import annotations

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


def _route(up: np.ndarray, down: np.ndarray, cfg) -> np.ndarray:
    """Route every scenario (row) of `up` into `down`, by travel time (n == 0) or Muskingum."""
    if cfg.n == 0:
        return np.vstack([propagate_tv(u, d, float(cfg.tv)) for u, d in zip(up, down)])
    return np.vstack([propagate_muskingum(u, d, int(cfg.n), cfg.c1, cfg.c2, cfg.c3) for u, d in zip(up, down)])


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
    posto_regredido, flag_regressao, posto_jusante, tv, n, c1, c2, c3).

    Returns data_rodada, cenario, data_previsao, nome (station), previsao_incremental, previsao_total,
    previsao_total_sem_tv (no routing delay) and previsao_inc_tv (total minus upstream totals).
    """
    if "variavel" in forecast:
        forecast = forecast[forecast["variavel"] == "Qcalc"]
    cfg = config.copy()
    for c in ("regressao", "posto_regredido", "flag_regressao", "posto_jusante", "tv", "n", "c1", "c2", "c3"):
        cfg[c] = pd.to_numeric(cfg[c]).fillna(0)
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
