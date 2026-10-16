from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysmap.routing import (muskingum_coeffs, propagate_muskingum, propagate_ssarr, propagate_todini, propagate_tv,
                            totalize)


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


def test_muskingum_coeffs_from_k_x_nf():
    c = muskingum_coeffs(2.77587067469559, 0.0178211110097375, 3)  # Lajeado, MPV validation report
    assert sum(c) == pytest.approx(1.0)
    up = np.r_[np.full(10, 100.0), np.full(30, 500.0)]
    out = propagate_muskingum(up, np.zeros(40), 3, *c)
    assert out[1:10] == pytest.approx(100.0) and out[-1] == pytest.approx(500.0, abs=1.0)
    with pytest.warns(UserWarning):
        muskingum_coeffs(1.0, 0.5, 1)


# Ts per phase must stay above ~12 h or MPV's daily scheme overshoots (24 / (Ts + 12) > 1).
TABLE_Q, TABLE_T = np.array([0.0, 1000.0, 5000.0, 20000.0]), np.array([96.0, 72.0, 48.0, 36.0])


def test_ssarr_steady_and_pulse():
    steady = propagate_ssarr(np.full(20, 800.0), np.zeros(20), 3, TABLE_Q, TABLE_T)
    np.testing.assert_allclose(steady, 800.0)
    up = np.full(120, 1000.0)
    up[10:13] = 6000.0
    out = propagate_ssarr(up, np.full(120, 5.0), 2, TABLE_Q, TABLE_T) - 5.0
    assert out.max() < up.max() and out.argmax() > up.argmax()  # attenuated and delayed
    assert out.sum() == pytest.approx(up.sum(), rel=1e-3)  # mass conserved (outputs are truncated)
    assert np.isfinite(propagate_ssarr(np.full(5, 30000.0), np.zeros(5), 2, TABLE_Q, TABLE_T)).all()  # above table


TODINI = (3010.590031, 8365.577144, 0.030098139, 0.04, 7.97520661157024e-5, 4.88790093712762, 120000.0)


def test_todini_steady_and_mass():
    np.testing.assert_allclose(propagate_todini(np.full(30, 3000.0), np.zeros(30), *TODINI), 3000.0)
    up = 2000.0 + 8000.0 * np.exp(-0.5 * ((np.arange(200) - 30) / 4.0) ** 2)  # flood crossing bankfull
    out = propagate_todini(up, np.zeros(200), *TODINI)
    assert out.max() < up.max()  # no visible delay: MPV reports O(t+1) on day t
    # MPV's variant (Co_t in C1, sum in D) loses ~0.9% of the flood volume on this case
    assert (out - 2000.0).sum() == pytest.approx((up - 2000.0).sum(), rel=0.02)


# Outputs of the real MPV binary (ONS.MPV.ConsoleApplication, 3-decimal files) on this 80-day flood, QD = 3.
MPV_INFLOW = np.round(2000.0 + 18000.0 * np.exp(-0.5 * ((np.arange(80) - 30) / 4.0) ** 2), 3)
MPV_CASES = {
    "sd13": lambda u, d: propagate_tv(u, d, 13.0),
    "sd36": lambda u, d: propagate_tv(u, d, 36.0),
    "sd48": lambda u, d: propagate_tv(u, d, 48.0),
    "mklaj": lambda u, d: propagate_muskingum(u, d, 3, *muskingum_coeffs(2.77587067469559, 0.0178211110097375, 3)),
    "mkband": lambda u, d: propagate_muskingum(u, d, 9, *muskingum_coeffs(24.9550439510318, 0.01, 9)),
    "ss2": lambda u, d: propagate_ssarr(u, d, 2, TABLE_Q, TABLE_T),
    "ss3": lambda u, d: propagate_ssarr(u, d, 3, TABLE_Q, TABLE_T),
    "todini": lambda u, d: propagate_todini(u, d, *TODINI),
}


@pytest.mark.parametrize("case", MPV_CASES)
def test_matches_mpv(case):
    g = pd.read_csv(Path(__file__).parent / "data" / "mpv_golden.csv").query("case == @case")
    out = MPV_CASES[case](MPV_INFLOW, np.zeros(80))
    np.testing.assert_allclose(out[g.day], g.mpv, atol=1e-3)


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


def test_totalize_mpv_methods(tmp_path):
    d = pd.Timestamp("2022-01-10")
    days = pd.date_range(d, periods=10)
    names = ("a", "b", "c", "e", "out")
    fc = pd.concat([pd.DataFrame({"data_caso": d, "data_previsao": days, "cenario": "c", "nome": b, "valor": 1000.0})
                    for b in names])
    obs = pd.concat([pd.DataFrame({"data": pd.date_range(d - pd.Timedelta(days=20), periods=20), "nome": b,
                                   "valor": 1000.0}) for b in names])
    pd.DataFrame({"vazao": TABLE_Q, "tempo": TABLE_T}).to_csv(tmp_path / "t.csv", sep=";", index=False)
    cfg = _config([(i + 1, b.upper(), b, "X", 1.0, None, None, None, 5 if i < 4 else 0, 12, 0, 0, 0, 0)
                   for i, b in enumerate(names)])
    cfg["metodo"] = ["SD", "muskingum", "ssarr", "todini", None]
    cfg["k"], cfg["x"], cfg["nf"] = 2.0, 0.2, 2
    cfg["tabela"] = str(tmp_path / "t.csv")
    for c, v in zip(("b1", "b2", "n1", "n2", "s0", "hi", "comprimento"), TODINI):
        cfg[c] = v
    out = totalize(fc, obs, cfg).set_index("nome")
    np.testing.assert_allclose(out.loc["OUT", "previsao_total"], 5000.0)  # steady flow through all 4 methods
    cfg["metodo"] = "nope"
    with pytest.raises(ValueError, match="metodo"):
        totalize(fc, obs, cfg)
