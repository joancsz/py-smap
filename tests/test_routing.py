import numpy as np
import pandas as pd
import pytest

from pysmap.routing import propagate_muskingum, propagate_tv, totalize


def test_travel_time_splits_over_two_days():
    up, down = np.arange(10.0), np.ones(10)
    out = propagate_tv(up, down, 36.0)  # 1.5 days: half from 1 day back, half from 2 days back
    np.testing.assert_allclose(out[2:], 0.5 * up[1:-1] + 0.5 * up[:-2] + 1)
    assert (out[:2] == 0).all()
    np.testing.assert_allclose(propagate_tv(up, down, 0.0), up + down)


def test_muskingum():
    up, down = np.arange(10.0), np.zeros(10)
    np.testing.assert_allclose(propagate_muskingum(up, down, 1, 0.0, 1.0, 0.0)[1:], up[:-1])  # pure 1-day lag
    const = propagate_muskingum(np.full(50, 7.0), down.repeat(5), 3, 0.2, 0.5, 0.3)
    np.testing.assert_allclose(const[1:], 7.0)  # c1+c2+c3 = 1 conserves a steady flow


def _config(rows):
    cols = ["posto", "nome_real", "sub_bacia_agrupada", "bacia_smap", "fator", "regressao", "posto_regredido",
            "flag_regressao", "posto_jusante", "tv", "n", "c1", "c2", "c3"]
    return pd.DataFrame(rows, columns=cols)


def test_totalize_network():
    d = pd.Timestamp("2022-01-10")
    days = pd.date_range(d, periods=5)
    fc = pd.concat([pd.DataFrame({"data_caso": d, "data_previsao": days, "cenario": c, "nome": b, "valor": v})
                    for c, k in (("c1", 1.0), ("c2", 2.0)) for b, v in (("up", 10 * k), ("down", 3 * k))])
    obs = pd.DataFrame({"data": pd.date_range(d - pd.Timedelta(days=5), periods=5), "nome": "up", "valor": 10.0})
    obs = pd.concat([obs, obs.assign(nome="down", valor=3.0)])
    cfg = _config([
        (1, "Up", "UP", "X", 1.0, None, None, None, 2, 0, 0, 0, 0, 0),
        (2, "Down", "DOWN", "X", 0.5, None, None, None, 0, 0, 0, 0, 0, 0),
        (3, "Side", "DOWN", "X", 0.5, 2.0, 1, 0, 0, 0, 0, 0, 0, 0),  # regressed on Up's total
    ])
    out = totalize(fc, obs, cfg).set_index(["cenario", "nome", "data_previsao"])
    c2 = out.loc["c2"]
    assert (c2.loc["Down", "previsao_incremental"] == 3.0).all()  # 6 * fator 0.5
    assert (c2.loc["Down", "previsao_total"] == 23.0).all()  # own 3 + Up's 20
    assert (c2.loc["Down", "previsao_inc_tv"] == 3.0).all()
    assert (c2.loc["Side", "previsao_incremental"] == 40.0).all()  # Up total * 2
    assert (c2.loc["Side", "previsao_total"] == 43.0).all()
    assert len(out) == 2 * 3 * 5


def test_totalize_flow_gauge_and_flag1_regression():
    d = pd.Timestamp("2022-01-10")
    fc = pd.concat([pd.DataFrame({"data_caso": d, "data_previsao": pd.date_range(d, periods=4), "cenario": "c",
                                  "nome": b, "valor": v}) for b, v in (("up", 20.0), ("down", 6.0))])
    obs = pd.DataFrame({"data": pd.date_range(d - pd.Timedelta(days=3), periods=3), "nome": "up", "valor": 20.0})
    obs = pd.concat([obs, obs.assign(nome="down", valor=6.0)])
    cfg = _config([
        (10, "Gauge", "UP", "posto_flu", 1.0, None, None, None, 2, 0, 0, 0, 0, 0),  # routes into Down's incremental
        (2, "Down", "DOWN", "X", 1.0, None, None, None, 0, 0, 0, 0, 0, 0),
        (4, "Reg", "DOWN", "X", 0.0, 0.5, 2, 1, 0, 0, 0, 0, 0, 0),  # flag 1: half of Down's own incremental
    ])
    out = totalize(fc, obs, cfg).set_index(["nome", "data_previsao"])
    assert set(out.index.get_level_values(0)) == {"Down", "Reg"}  # flow gauges are not reported
    assert (out.loc["Down", "previsao_incremental"] == 26.0).all()
    assert (out.loc["Down", "previsao_total"] == 26.0).all()
    assert (out.loc["Reg", "previsao_incremental"] == 3.0).all()  # taken before the gauge routing, as in R
