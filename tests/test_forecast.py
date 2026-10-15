import time

import numpy as np
import pandas as pd
import pytest

from pysmap import forecast, weighting
from pysmap.forecast import Inputs, _init_value, run_chained
from pysmap.io import fill_forecast, history_as_forecast

BASINS = ["avermelha", "ssimao2"]
DATES = pd.DataFrame({"data": pd.to_datetime(["2020-05-01", "2020-05-08"]), "numero_dias_previsao": [15, 20]})


def two_scenarios(obs, dates, first_day=1, extra=2):
    """History replayed as forecast, plus an identical 'cenario2' (R's CT25 setup)."""
    one = history_as_forecast(obs, dates, first_day, extra)
    return pd.concat([one, one.assign(cenario="cenario2")], ignore_index=True)


def ct25_inputs(rdata, **etp):
    gauges = rdata("postos_plu").query("nome in @BASINS")
    prec = weighting.spatial(rdata("historico_precipitacao"), gauges)
    vals = {"avermelha": [218.71, 46.69, 0.2891, 31], "ssimao2": [441.67, 256.98, 0.3141, 31]}
    init = pd.DataFrame([(b, v, x, 0) for b, xs in vals.items()
                         for v, x in zip(["Ebin", "Supin", "Tuin", "numero_dias_assimilacao"], xs)]
                        + [(b, v, x, m) for b in BASINS for m in range(1, 13)
                           for v, x in [("limite_inferior_ebin", 0.8), ("limite_superior_ebin", 1.2),
                                        ("limite_superior_prec", 2), ("limite_inferior_prec", 0.5)]],
                        columns=["nome", "variavel", "valor", "mes"])
    flow = rdata("historico_vazao").rename(columns={"posto": "nome"})
    return Inputs(parameters=rdata("parametros").query("nome in @BASINS"), initialization=init,
                  precip_observed=prec, precip_forecast=two_scenarios(prec, DATES), flow_observed=flow,
                  run_dates=DATES, sub_basins=BASINS, **etp)


def check_ct25(out):
    prev = out["previsao"]
    day = prev[prev.data_previsao == "2020-05-05"]
    np.testing.assert_array_equal(day[day.cenario == "cenario2"].valor.to_numpy(),
                                  day[day.cenario == "historico"].valor.to_numpy())
    # CT25.2: the second date hot-starts from the first date's assimilation
    opt = out["otimizacao"]
    ebin2 = opt[(opt.nome == "ssimao2") & (opt.data_caso == "2020-05-08") & (opt.variavel == "Ebin")]
    a = out["assimilacao"]
    qbase = a[(a.nome == "ssimao2") & (a.data_caso == "2020-05-01") & (a.variavel == "Qbase")
              & (a.data_assimilacao == pd.Timestamp("2020-05-08") - pd.Timedelta(days=32))]
    assert ebin2.limite_inferior.item() / 0.8 == pytest.approx(qbase.valor.item())


def test_chained_climatology(rdata):  # test-rodadas_encadeadas.R CT25.1/CT25.2
    inputs = ct25_inputs(rdata, etp_climatology=rdata("historico_etp_NC"))
    t = time.perf_counter()
    out = run_chained(inputs, forecast.OFICIAL)
    print(f"run_chained: {time.perf_counter() - t:.1f}s")
    check_ct25(out)
    assert np.isfinite(out["funcao_objetivo"].funcao_objetivo).all()
    assert "Ai" not in set(out["previsao"].variavel)


def test_chained_etp_series(rdata):  # "testa rodada com serie temporal etp"
    etp = rdata("historico_etp").rename(columns={"posto": "nome"})
    fc = two_scenarios(etp, DATES, first_day=0, extra=-1)
    out = run_chained(ct25_inputs(rdata, etp_observed=etp, etp_forecast=fc), forecast.ETP)
    check_ct25(out)
    assert out["otimizacao"].variavel.str.startswith("etp").any()


def test_open_loop_keeps_initial_state(rdata):
    out = run_chained(ct25_inputs(rdata, etp_climatology=rdata("historico_etp_NC")),
                      forecast.RunConfig(assimilate=False))
    opt = out["otimizacao"]
    first = opt[(opt.nome == "avermelha") & (opt.data_caso == "2020-05-01")].set_index("variavel").otimizacao
    assert first["Ebin"] == 218.71 and first["Supin"] == 46.69


def test_fill_forecast_adds_only_missing_days_with_run_mean():
    d1, d2 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-08")
    fc = pd.DataFrame({"data_rodada": [d1] * 3 + [d2] * 5, "nome": "A", "cenario": "c",
                       "data_previsao": list(pd.date_range(d1 + pd.Timedelta(days=1), periods=3))
                       + list(pd.date_range(d2 + pd.Timedelta(days=1), periods=5)),
                       "valor": [1.0, 2.0, 3.0, 10, 10, 10, 10, 10]})[
        ["data_rodada", "data_previsao", "cenario", "nome", "valor"]]
    dates = pd.DataFrame({"data": [d1, d2], "numero_dias_previsao": [4, 4]})
    out = fill_forecast(fc, dates, extra_days=2)  # needs d+1 .. d+5
    r1 = out[out.data_rodada == d1].set_index("data_previsao").valor
    assert list(r1) == [1, 2, 3, 2, 2]  # two missing days filled with this run's mean
    assert len(out[out.data_rodada == d2]) == 5  # complete run untouched


def test_month_bounds_never_borrow_another_month():
    init = pd.DataFrame({"nome": "A", "variavel": "limite_superior_prec", "valor": [2.0, 1.5], "mes": [1, 6]})
    assert _init_value(init, "A", "limite_superior_prec", 6) == 1.5
    with pytest.raises(ValueError, match="missing"):
        _init_value(init, "A", "limite_superior_prec", 7)
    init0 = pd.concat([init, pd.DataFrame({"nome": ["A"], "variavel": "limite_superior_prec", "valor": 1.8, "mes": 0})])
    assert _init_value(init0, "A", "limite_superior_prec", 7) == 1.8


def test_missing_climatology_month_fails(rdata):
    nc = rdata("historico_etp_NC").query("mes != 4")  # the May runs assimilate April
    with pytest.raises(ValueError, match=r"lacks months \[4\]"):
        run_chained(ct25_inputs(rdata, etp_climatology=nc), forecast.OFICIAL)


def test_workers_give_the_same_result(rdata):
    inputs = ct25_inputs(rdata, etp_climatology=rdata("historico_etp_NC"))
    a, b = run_chained(inputs, forecast.OFICIAL), run_chained(inputs, forecast.OFICIAL, workers=2)
    pd.testing.assert_frame_equal(a["previsao"], b["previsao"])
