"""Chained forecast runs: for each sub-basin and forecast date, assimilate then simulate every scenario.

`run_chained` replaces smapOnsR's rodada_encadeada_{oficial,etp,pmur,pmur_etp,pmur_nAssimilaEtp} and
rodada_sem_assimilacao; `RunConfig` picks the variant.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import weighting
from .assimilation import assimilate
from .kernel import COL, COLUMNS, simulate
from .params import SmapParams, initial_state

DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class RunConfig:
    runoff: Literal["ai", "pmur"] = "ai"
    etp_source: Literal["climatology", "series"] = "climatology"
    assimilate_etp: bool = False  # only with etp_source="series"
    assimilate: bool = True  # False: open-loop run from the given initial state

    def __post_init__(self):
        if self.assimilate_etp and self.etp_source != "series":
            raise ValueError("assimilate_etp needs etp_source='series'")


# R presets, by the name of the function they replace.
OFICIAL = RunConfig()
ETP = RunConfig(etp_source="series", assimilate_etp=True)
PMUR = RunConfig(runoff="pmur")
PMUR_ETP = RunConfig(runoff="pmur", etp_source="series", assimilate_etp=True)
PMUR_N_ASSIMILA_ETP = RunConfig(runoff="pmur", etp_source="series")


@dataclass
class Inputs:
    """Model inputs as long DataFrames. Observed series use columns `data`, `nome`, `valor`;
    forecasts use `data_rodada`, `data_previsao`, `cenario`, `nome`, `valor`."""
    parameters: pd.DataFrame  # nome, parametro, valor
    initialization: pd.DataFrame  # nome, variavel, valor[, mes]; Tuin as a fraction
    precip_observed: pd.DataFrame  # per sub-basin, already spatially weighted
    precip_forecast: pd.DataFrame
    flow_observed: pd.DataFrame
    run_dates: pd.DataFrame  # data, numero_dias_previsao
    sub_basins: list[str]
    etp_climatology: pd.DataFrame | None = None  # mes, nome, valor
    etp_observed: pd.DataFrame | None = None
    etp_forecast: pd.DataFrame | None = None


def _init_value(init: pd.DataFrame, name: str, var: str, month: int | None = None, default=None) -> float:
    sub = init[(init["nome"] == name) & (init["variavel"] == var)]
    if month is not None and "mes" in sub:
        # the month's own row, else an all-months row (mes 0); never another month's value
        sub = sub[sub["mes"] == month] if (sub["mes"] == month).any() else sub[sub["mes"] == 0]
    if sub.empty:
        if default is None:
            raise ValueError(f"sub-basin {name}: missing initialization variable {var}")
        return default
    return float(sub["valor"].iloc[0])


def _window(series: pd.Series, start, end, what: str) -> np.ndarray:
    """Values for every day in [start, end]; fails loudly on gaps."""
    idx = pd.date_range(start, end)
    out = series.reindex(idx)
    if out.isna().any():
        raise ValueError(f"{what}: no data for {list(idx[out.isna()].date[:5])}")
    return out.to_numpy()


def _climatology(nc: pd.Series, days: pd.DatetimeIndex, what: str) -> np.ndarray:
    """Monthly ETP normals for each day; fails loudly on a missing month."""
    out = nc.reindex(days.month).to_numpy()
    if np.isnan(out).any():
        raise ValueError(f"{what}: ETP climatology lacks months {sorted(set(days.month[np.isnan(out)]))}")
    return out


def _scenario_table(obs: pd.Series, forecast: pd.DataFrame, what: str) -> pd.DataFrame:
    """Dates x scenarios: the observed series repeated for every scenario, then each scenario's forecast."""
    fc = forecast.pivot(index="data_previsao", columns="cenario", values="valor").sort_index(axis=1)
    if fc.empty:
        raise ValueError(f"{what}: no forecast")
    obs_part = pd.DataFrame(np.repeat(obs.to_numpy()[:, None], fc.shape[1], axis=1), index=obs.index,
                            columns=fc.columns)
    table = pd.concat([obs_part, fc])
    if table.index.duplicated().any():
        raise ValueError(f"{what}: observed and forecast overlap on {table.index[table.index.duplicated()][0]}")
    return table


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    sub = df[df["nome"] == name]
    return pd.Series(sub["valor"].to_numpy(), index=pd.DatetimeIndex(sub["data"])).sort_index()


def _check(inputs: Inputs, config: RunConfig):
    if config.etp_source == "series" and (inputs.etp_observed is None or inputs.etp_forecast is None):
        raise ValueError("etp_source='series' needs etp_observed and etp_forecast")
    if config.etp_source == "climatology" and inputs.etp_climatology is None:
        raise ValueError("etp_source='climatology' needs etp_climatology")


def iter_sub_basins(inputs: Inputs, config: RunConfig = OFICIAL, workers: int | None = None):
    """Yield (sub-basin, tables) as each sub-basin's chain finishes; see `run_chained` for the tables.

    Sub-basins are independent, so `workers` > 1 runs them in parallel processes (order kept).
    Consuming this one sub-basin at a time keeps memory flat on large runs.
    """
    _check(inputs, config)
    if not workers or workers < 2:
        for name in inputs.sub_basins:
            yield name, run_sub_basin(inputs, config, name)
        return
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(inputs, config)) as pool:
        yield from zip(inputs.sub_basins, pool.map(_worker_run, inputs.sub_basins))


_WORKER: tuple = ()


def _worker_init(inputs, config):  # each process receives the inputs once, not once per sub-basin
    global _WORKER
    _WORKER = (inputs, config)


def _worker_run(name):
    return run_sub_basin(*_WORKER, name)


def run_chained(inputs: Inputs, config: RunConfig = OFICIAL, workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Run every sub-basin and date. Returns R's output tables (long format): previsao (data_caso,
    data_previsao, cenario, nome, variavel, valor), otimizacao, funcao_objetivo and assimilacao."""
    parts = [t for _, t in iter_sub_basins(inputs, config, workers)]
    return {k: pd.concat([t[k] for t in parts], ignore_index=True) for k in parts[0]}


def run_sub_basin(inputs: Inputs, config: RunConfig, name: str) -> dict[str, pd.DataFrame]:
    """All forecast dates of one sub-basin, chained by hot start."""
    _check(inputs, config)
    pmur = config.runoff == "pmur"
    series_etp = config.etp_source == "series"
    columns = list(COLUMNS if pmur else COLUMNS[:-1])  # Ai is constant in the classic model
    dates = inputs.run_dates.sort_values("data").reset_index(drop=True)
    init = inputs.initialization
    out = {k: [] for k in ("previsao", "otimizacao", "funcao_objetivo", "assimilacao")}
    p = SmapParams.from_table(inputs.parameters, name)
    ebin = _init_value(init, name, "Ebin")
    supin = _init_value(init, name, "Supin")
    tuin = _init_value(init, name, "Tuin")
    n = int(_init_value(init, name, "numero_dias_assimilacao"))
    objective = int(_init_value(init, name, "funcao_objetivo", default=0))
    adjust = _init_value(init, name, "ajusta_precipitacao", default=0) == 1
    prec_obs = _series(inputs.precip_observed, name)
    flow_obs = _series(inputs.flow_observed, name)
    etp_obs = _series(inputs.etp_observed, name) if series_etp else None
    if not series_etp:
        nc = inputs.etp_climatology[inputs.etp_climatology["nome"] == name].set_index("mes")["valor"]
    prec_fc = inputs.precip_forecast[inputs.precip_forecast["nome"] == name]
    etp_fc = inputs.etp_forecast[inputs.etp_forecast["nome"] == name] if series_etp else None

    for i, (d, n_fc) in enumerate(zip(dates["data"], dates["numero_dias_previsao"].astype(int))):
        d = pd.Timestamp(d)
        what = f"{name} {d.date()}"
        q = _window(flow_obs, d - n * DAY, d - DAY, f"{what} observed flow")
        obs = prec_obs[(prec_obs.index >= d - (n + p.kt_min) * DAY) & (prec_obs.index <= d)]
        rain = _scenario_table(obs, prec_fc[prec_fc["data_rodada"] == d], f"{what} rain")
        scenarios = list(rain.columns)
        first = rain.iloc[:, 0]

        # Assimilation window: days d-n .. d-1.
        prec_assim = _window(first, d - (n + p.kt_min) * DAY, d + (p.kt_max - 1) * DAY, f"{what} rain")
        if series_etp:
            etp_assim = _window(etp_obs, d - n * DAY, d - DAY, f"{what} observed ETP")
        else:
            etp_assim = _climatology(nc, pd.date_range(d - n * DAY, d - DAY), what)
        month = d.month
        a = assimilate(
            p, tuin, ebin, supin, prec_assim, etp_assim * p.ecof, etp_assim * p.ecof2, q, pmur=pmur,
            assimilate_etp=config.assimilate_etp,
            prec_bounds=(_init_value(init, name, "limite_inferior_prec", month),
                         _init_value(init, name, "limite_superior_prec", month)),
            ebin_bounds=(_init_value(init, name, "limite_inferior_ebin", month),
                         _init_value(init, name, "limite_superior_ebin", month)),
            objective=objective, adjust_last_day=adjust, fit=config.assimilate)

        # Forecast window: days d-n .. d+n_fc-1, re-run from the assimilated initial state.
        # As in R, rain weights apply to days d-n .. d-2; day d-1 is recomputed with the raw rain
        # (scaled by min(w[d-2], 1) when ajusta_precipitacao), the forecast part's first day.
        r = rain * p.pcof
        past = r.loc[d - (n + p.kt_min) * DAY: d + (p.kt_max - 2) * DAY].to_numpy().T
        past = weighting.temporal(past, p.kt, p.kt_max, p.kt_min) * a.prec_weights[:n - 1]
        if adjust:
            r.loc[:d] *= min(a.prec_weights[n - 2], 1.0)
        future = r.loc[d - (p.kt_min + 1) * DAY: d + (n_fc + p.kt_max - 1) * DAY].to_numpy().T
        future = weighting.temporal(future, p.kt, p.kt_max, p.kt_min)
        prec = np.hstack([past, future])
        if np.isnan(prec).any():
            raise ValueError(f"{what}: rain forecast has gaps")
        if prec.shape[1] != n + n_fc:
            raise ValueError(f"{what}: rain covers {prec.shape[1]} of {n + n_fc} days")

        if series_etp:
            fc = etp_fc[(etp_fc["data_rodada"] == d) & (etp_fc["data_previsao"] < d + n_fc * DAY)]
            obs_etp = etp_obs[(etp_obs.index >= d - n * DAY) & (etp_obs.index < d)]
            e = _scenario_table(obs_etp, fc, f"{what} ETP")
            missing = set(scenarios) - set(e.columns)
            if missing:
                raise ValueError(f"{what}: no ETP forecast for scenarios {sorted(missing)[:5]}")
            etp = np.vstack([_window(e[s], d - n * DAY, d + (n_fc - 1) * DAY, f"{what} ETP {s}")
                             for s in scenarios])
            if a.etp_weights is not None:
                etp[:, :n] *= a.etp_weights
        else:
            etp = np.broadcast_to(_climatology(nc, pd.date_range(d - n * DAY, periods=n + n_fc), what),
                                  prec.shape)
        s0 = initial_state(p, a.ebin, tuin, a.supin)
        sim = simulate(p, s0, prec, etp * p.ecof, etp * p.ecof2, pmur)[:, n:, :len(columns)]

        fc_dates = pd.date_range(d, periods=n_fc)
        prev = pd.DataFrame(sim.reshape(-1, len(columns)), columns=columns)
        prev.insert(0, "data_caso", d)
        prev.insert(1, "data_previsao", np.tile(fc_dates, len(scenarios)))
        prev.insert(2, "cenario", np.repeat(scenarios, n_fc))
        prev.insert(3, "nome", name)
        out["previsao"].append(prev)

        assim = pd.DataFrame(a.simulation[:, :len(columns)], columns=columns)
        assim.insert(0, "data_caso", d)
        assim.insert(1, "data_assimilacao", pd.date_range(d - n * DAY, periods=n))
        assim.insert(2, "nome", name)
        out["assimilacao"].append(assim)
        out["otimizacao"].append(a.table.assign(nome=name, data_caso=d))
        out["funcao_objetivo"].append(pd.DataFrame({"funcao_objetivo": [a.objective], "nome": name,
                                                    "data_caso": d}))

        if i + 1 < len(dates):  # hot start: state the day before the next assimilation window
            t = (pd.Timestamp(dates["data"][i + 1]) - (n + 1) * DAY - (d - n * DAY)).days
            if not 0 <= t < n:
                raise ValueError(f"{name}: next date {dates['data'][i + 1]} is outside the hot-start "
                                 f"range of {d.date()} (assimilation of {n} days)")
            row = a.simulation[t]
            ebin = row[COL["Qbase"]]
            supin = row[COL["Qsup1"]] + row[COL["Qsup2"]] + row[COL["Qplan"]]
            tuin = row[COL["Tu"]]

    res = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    res["previsao"] = res["previsao"].melt(id_vars=["data_caso", "data_previsao", "cenario", "nome"],
                                           var_name="variavel", value_name="valor")
    res["assimilacao"] = res["assimilacao"].melt(id_vars=["data_caso", "data_assimilacao", "nome"],
                                                 var_name="variavel", value_name="valor")
    return res
