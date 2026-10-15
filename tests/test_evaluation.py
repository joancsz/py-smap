import pandas as pd
import pytest

from conftest import DATA
from pysmap.evaluation import evaluate
from pysmap.io import _wide_history


def _get(out, level, h, metric):
    return out.query("discretizacao == @level and horizonte == @h and metrica == @metric").valor.item()


def test_real_forecasts():  # test-metricas.R "testa previsoes reais"
    prev = pd.read_csv(DATA / "arq_saida/previsao.csv", sep=";", parse_dates=["data_caso", "data_previsao"])
    obs = _wide_history(DATA / "arq_saida/vazao_observada.csv")
    out = evaluate(prev, obs)
    assert _get(out, "diaria", 8, "PBIAS") == pytest.approx(0.97653114, rel=1e-7)
    assert _get(out, "semanal", 3, "DM") == pytest.approx(0.40269503, rel=1e-7)
    assert _get(out, "mensal", 1, "KGE") == pytest.approx(0.9217572, rel=1e-7)


def _sim(cases, values):
    return pd.concat([pd.DataFrame({"data_caso": d, "data_previsao": d + pd.to_timedelta(range(len(values)), "D"),
                                    "cenario": "h", "nome": "A", "variavel": "Qcalc", "valor": values})
                      for d in cases], ignore_index=True)


def test_daily_two_runs():  # test-metricas.R horizon-1 case
    cases = pd.to_datetime(["2000-01-01", "2000-01-10"])
    obs = pd.DataFrame({"data": [c + pd.Timedelta(days=i) for c in cases for i in range(3)], "nome": "A",
                        "valor": [12, 18, 25, 11, 17, 24]})
    out = evaluate(_sim(cases, [10, 20, 30]), obs, weekly=False, monthly=False)
    assert len(out) == 18
    assert _get(out, "diaria", 1, "PBIAS") == pytest.approx(1 + (40 - 35) / 35)


def test_first_horizons():
    d = pd.Timestamp("2000-01-01")
    obs = pd.DataFrame({"data": pd.date_range(d, periods=5), "nome": "A", "valor": range(1, 6)})
    out = evaluate(_sim([d], list(range(1, 6))), obs)
    assert out.groupby("discretizacao", sort=False).horizonte.first().to_dict() == {"diaria": 0, "semanal": 1,
                                                                                     "mensal": 1}


def test_monthly_blocks_use_only_observed_days():
    d = pd.Timestamp("2000-01-01")
    obs = pd.DataFrame({"data": [d, d + pd.Timedelta(days=2)], "nome": "A", "valor": [10.0, 30.0]})
    out = evaluate(_sim([d], [10.0, 100.0, 30.0]), obs, weekly=False)
    assert _get(out, "mensal", 1, "PBIAS") == pytest.approx(1.0)  # day 2 has no observation, so it is left out
