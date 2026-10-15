"""Temporal (Kt) and spatial (rain-gauge) weighting of precipitation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def temporal(x, kt: np.ndarray, kt_max: int, kt_min: int) -> np.ndarray:
    """Weighted moving sum over the last axis; output loses kt_max + kt_min points.

    Output i covers inputs i .. i+kt_min+kt_max: kt_min past days, the day itself, kt_max future days.
    Keeps R's formula sum(w*x)/sum(w)*sum(kt), which only differs from sum(w*x) when kt has
    weight outside the window.
    """
    w = kt[2 - kt_max: 3 + kt_min][::-1]  # oldest lag first
    win = sliding_window_view(np.asarray(x, dtype=float), len(w), axis=-1)
    return win @ w / w.sum() * kt.sum()


def spatial(precip: pd.DataFrame, gauges: pd.DataFrame, by=("data",)) -> pd.DataFrame:
    """Weighted sum of gauge series into one series per sub-basin.

    precip: columns `posto`, `valor` plus the `by` keys; gauges: `nome`, `posto`, `valor` (weight).
    Returns the `by` keys, `nome` and `valor`.
    """
    m = precip.merge(gauges.rename(columns={"valor": "peso"}), on="posto")
    m["valor"] = m["valor"] * m["peso"]
    return m.groupby([*by, "nome"], as_index=False, sort=True)["valor"].sum()
