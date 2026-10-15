import numpy as np
import pandas as pd
import pytest

from pysmap.calibration import calibrate, make_kt
from pysmap.params import SmapParams


def test_make_kt_is_a_normalized_window():
    kt = make_kt(1, 4, 5, 0.5)
    assert kt.sum() == pytest.approx(1)
    assert (kt[1:7] > 0).all() and (kt[:1] == 0).all() and (kt[7:] == 0).all()


def _case(rdata):  # test-calibracao.R CT22.1, on a shorter period
    p = SmapParams.from_table(rdata("parametros"), "baixoig")
    gauges = rdata("postos_plu").query("nome == 'baixoig'")
    rain = rdata("historico_precipitacao").query("posto in @gauges.posto")
    start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31")
    nc = rdata("historico_etp_NC").query("nome == 'baixoig'").set_index("mes").valor
    days = pd.date_range(start, end)
    etp = pd.Series(nc.reindex(days.month).to_numpy(), index=days)
    flow = rdata("historico_vazao").query("posto == 'baixoig'").set_index("data").valor
    bounds = {f: (getattr(p, f) * 0.01, getattr(p, f) * 10)
              for f in ("str_", "k2t", "crec", "capc", "k_kt", "h1", "k2t2", "h", "k1t", "k3t")}
    bounds |= {"pcof": (0.8, 1.2), "ecof": (0.8, 1.2), "ecof2": (0.8, 1.2), "alfa": (1e-5, 0.99), "ex": (1e-5, 0.99)}
    return dict(p=p, rain=rain, gauges=gauges, etp=etp, flow=flow, start=start, end=end, ebin=300, tuin=0.5,
                supin=700, kt_shape=(5, 0.5)), bounds


def test_calibration_improves_the_fit(rdata):
    case, bounds = _case(rdata)
    period = (pd.Timestamp("2019-01-01"), case["end"])
    before = calibrate(**case, bounds={}, objective_period=period).objective
    res = calibrate(**case, bounds=bounds, objective_period=period, maxiter=30)
    assert res.objective < before
    assert res.params.kt.sum() == pytest.approx(1)


def test_zero_weights_equal_a_shorter_period(rdata):  # CT22.2
    case, _ = _case(rdata)
    period = (pd.Timestamp("2019-01-01"), case["end"])
    short = calibrate(**case, bounds={}, objective_period=period).objective
    days = pd.date_range(case["start"], case["end"])
    w = np.where(days >= period[0], 1.0, 0.0)
    full = calibrate(**case, bounds={}, weights=w / w.sum()).objective
    assert full == pytest.approx(short)


def test_zero_flow_days_keep_the_objective_finite(rdata):
    case, _ = _case(rdata)
    case["flow"] = case["flow"].where(case["flow"].index.month != 9, 0.0)  # a dry month
    assert np.isfinite(calibrate(**case, bounds={}).objective)
