"""Forecast skill by lead time: daily, weekly (Saturday-start) and 30-day blocks (R's analisa_previsoes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import dm, kge, mape, nse, pbias, rmse

METRICS = {"PBIAS": pbias, "NSE": nse, "MAPE": mape, "DM": dm, "KGE": kge, "RMSE": rmse}
KEYS = ["data_caso", "cenario", "nome"]


def _score(df: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):  # one-point groups give NaN, as in R
        for (name, h), g in df.groupby(["nome", "horizonte"]):
            sim, obs = g["prev"].to_numpy(), g["obs"].to_numpy()
            rows += [(name, m, f(sim, obs), level, h) for m, f in METRICS.items()]
    return pd.DataFrame(rows, columns=["nome", "metrica", "valor", "discretizacao", "horizonte"])


def evaluate(forecast: pd.DataFrame, observed: pd.DataFrame, weekly: bool = True, monthly: bool = True):
    """Score forecasts against observations per lead time.

    forecast: `previsao` rows (data_caso, data_previsao, cenario, nome[, variavel], valor); only Qcalc is used.
    observed: data, nome, valor. Each scenario counts as one forecast. Returns nome, metrica, valor,
    discretizacao (diaria/semanal/mensal), horizonte (days from 0; weeks and 30-day blocks from 1).
    """
    f = forecast[forecast["variavel"] == "Qcalc"] if "variavel" in forecast else forecast
    f = f[[*KEYS, "data_previsao", "valor"]].rename(columns={"valor": "prev"})
    f["horizonte"] = (f["data_previsao"] - f["data_caso"]).dt.days
    obs = observed.rename(columns={"data": "data_previsao", "valor": "obs"})[["data_previsao", "nome", "obs"]]
    daily = f.merge(obs, on=["data_previsao", "nome"])
    parts = [_score(daily, "diaria")]

    if weekly:  # 7-day blocks starting on each run's first Saturday; the last block may be partial
        sat = daily[daily["data_previsao"].dt.dayofweek == 5].groupby(KEYS)["data_previsao"].min()
        w = daily.join(sat.rename("inicio"), on=KEYS)
        w = w[w["data_previsao"] >= w["inicio"]]
        w["horizonte"] = (w["data_previsao"] - w["inicio"]).dt.days // 7 + 1
        parts.append(_score(w.groupby([*KEYS, "horizonte"], as_index=False)[["prev", "obs"]].mean(), "semanal"))

    if monthly:  # 30-day blocks over the days that have both values, so both means cover the same days
        m = daily.assign(horizonte=daily["horizonte"] // 30 + 1)
        parts.append(_score(m.groupby([*KEYS, "horizonte"], as_index=False)[["prev", "obs"]].mean(), "mensal"))
    return pd.concat(parts, ignore_index=True)
