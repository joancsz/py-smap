import numpy as np
import pandas as pd
import pytest

from pysmap import metrics, weighting
from pysmap.kernel import COL, simulate
from pysmap.params import SmapParams, initial_state


def test_kernel_two_days_sobradinho(rdata):  # test-smap_ons.R
    p = SmapParams.from_table(rdata("parametros"), "sobradinho")
    s0 = initial_state(p, ebin=800, tuin=0.15, supin=300)
    prec, etp, emarg = [0.846906, 0.904504], [5.29 * 0.9] * 2, [5.29] * 2
    out = simulate(p, s0, prec, etp, emarg)
    np.testing.assert_allclose(out[:, COL["Qcalc"]], [1100, 1065.9265], rtol=1e-8)
    two = simulate(p, s0, [prec, prec], [etp, etp], [emarg, emarg])
    np.testing.assert_array_equal(two[0], out)
    np.testing.assert_array_equal(two[1], out)


def test_initial_state_validates(rdata):
    p = SmapParams.from_table(rdata("parametros"), "sobradinho")
    with pytest.raises(ValueError):
        initial_state(p, ebin=1, tuin=1.5, supin=1)


def test_pmur_mode_needs_its_parameters(rdata):
    p = SmapParams.from_table(rdata("parametros"), "sobradinho")
    with pytest.raises(ValueError, match="PMUR"):
        simulate(p, initial_state(p, 800, 0.15, 300), [1.0], [1.0], [1.0], pmur=True)


def test_temporal_weighting_avermelha(rdata):  # test-ponderacao.R CT24.1/CT24.3
    p = SmapParams.from_table(rdata("parametros"), "avermelha")
    prec = rdata("historico_precipitacao")
    ini, fim = pd.Timestamp("2021-12-01"), pd.Timestamp("2021-12-29")
    x = prec[(prec.posto == "psatagv") & (prec.data >= ini - pd.Timedelta(days=p.kt_min))
             & (prec.data <= fim + pd.Timedelta(days=p.kt_max))].sort_values("data").valor.to_numpy()
    assert weighting.temporal(x, p.kt, p.kt_max, p.kt_min)[28] == pytest.approx(4.9405, rel=1e-8)
    kt = p.kt.copy()
    kt[1] += 0.001
    assert weighting.temporal(x, kt, p.kt_max, p.kt_min)[28] == pytest.approx(4.95354, rel=1e-8)
    two = weighting.temporal(np.vstack([x, 2 * x]), kt, p.kt_max, p.kt_min)
    assert two[1, 28] == pytest.approx(2 * 4.95354, rel=1e-8)


def test_spatial_weighting_pimentalt(rdata):  # test-ponderacao.R CT24.2
    gauges = rdata("postos_plu").query("nome == 'pimentalt'")
    prec = rdata("historico_precipitacao")
    day = pd.Timestamp("2021-12-30")
    sub = prec[(prec.data == day) & prec.posto.isin(gauges.posto)]
    out = weighting.spatial(sub, gauges)
    expected = sub.set_index("posto").valor[gauges.posto].to_numpy() @ gauges.valor.to_numpy()
    assert out.valor.item() == pytest.approx(expected)


def test_metrics():  # test-metricas.R CT23
    obs = np.arange(1, 31, dtype=float)
    sim = obs - 0.5
    assert metrics.nse(sim, obs) == pytest.approx(0.99666296, rel=1e-7)
    assert metrics.mape(sim, obs) == pytest.approx(0.066583119, rel=1e-7)
    assert metrics.dm(sim, obs) == pytest.approx(0.06666669, rel=1e-7)
    assert metrics.kge(sim, obs) == pytest.approx(0.96774194, rel=1e-7)
    assert metrics.pbias(sim, obs) == pytest.approx(0.96774194, rel=1e-7)
    assert metrics.correlation(sim, obs) == pytest.approx(1)
