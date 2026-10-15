import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from conftest import DATA
from pysmap.case import load_case
from pysmap.cli import main
from pysmap.metrics import OBJECTIVES, kge
from pysmap.studies import calibrate_case, thursdays

CALIBRATION = {"simulation": {"start": "2015-06-01", "end": "2016-12-31"},
               "objective": {"start": "2016-01-01", "end": "2016-12-31"},
               "exclude": [["2016-03-01", "2016-03-20"]], "maxiter": 15, "horizon": 14}


def _case(tmp_path, edit=lambda r: None):
    raw = yaml.safe_load((DATA / "case_pmur.yml").read_text())
    raw["data"] = {k: str(DATA / v) for k, v in raw["data"].items()}
    raw["calibration"] = dict(CALIBRATION)
    raw["sub_basins"]["avermelha"]["kt_shape"] = {"alfa": 5, "ex": 0.5}
    raw["sub_basins"]["avermelha"]["calibration_bounds"] = {
        "str": {"times": [0.5, 2]}, "k2t": [1, 50], "pcof": [0.8, 1.2], "alfa": [0.5, 10], "ex": [0.05, 0.95]}
    edit(raw)
    p = tmp_path / "case.yml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_kge_objective():
    assert OBJECTIVES[3] == (kge, -1)


def test_thursdays():
    weekly = thursdays("2016-01-01", "2016-02-01", 42)  # 2016-01-01 is a Friday
    assert weekly[0] == pd.Timestamp("2016-01-07") and (weekly.dayofweek == 3).all()
    assert (weekly[1:] - weekly[:-1] == pd.Timedelta(days=7)).all()
    monthly = thursdays("2016-01-01", "2016-04-30", 45)
    assert list(monthly) == list(pd.to_datetime(["2016-01-07", "2016-02-07", "2016-03-07", "2016-04-07"]))


def test_calibrate_case_and_round_trip(tmp_path):
    path = _case(tmp_path)
    main(["calibrate", str(path), "-o", str(tmp_path / "cal")])
    summary = pd.read_csv(tmp_path / "cal/calibracao_resumo.csv", sep=";").set_index("nome")
    assert list(summary.index) == ["avermelha"]
    assert summary.loc["avermelha", "objetivo_final"] < summary.loc["avermelha", "objetivo_inicial"]
    serie = pd.read_csv(tmp_path / "cal/calibracao_serie.csv", sep=";", parse_dates=["data"])
    excluded = serie[(serie.data >= "2016-03-01") & (serie.data <= "2016-03-20")]
    assert (excluded.peso == 0).all() and (serie[serie.data < "2016-01-01"].peso == 0).all()
    assert serie.peso.sum() == pytest.approx(1)
    log = pd.read_csv(tmp_path / "cal/calibracao_log.csv", sep=";")
    assert len(log) >= 1 and {"str_", "alfa", "objetivo"} <= set(log.columns)

    res = calibrate_case(path)["avermelha"]  # deterministic: same as the CLI run
    calibrated = load_case(tmp_path / "cal/case_calibrated.yml", run_dates=pd.DataFrame(
        {"data": [pd.Timestamp("2017-06-01")], "numero_dias_previsao": [42]}))
    p = calibrated.inputs.parameters.query("nome == 'avermelha'").set_index("parametro").valor
    assert p["Str"] == pytest.approx(res.params.str_) and p["K2t"] == pytest.approx(res.params.k2t)
    assert 0.5 * 154.5055324 <= res.params.str_ <= 2 * 154.5055324  # relative bound resolved


@pytest.mark.parametrize("edit, match", [
    (lambda r: r["sub_basins"]["avermelha"].pop("kt_shape"), "alfa/ex need kt_shape"),
    (lambda r: r["calibration"].update(objective={"start": "2014-01-01", "end": "2016-12-31"}), "inside"),
    (lambda r: r["sub_basins"]["avermelha"]["calibration_bounds"].update(foo=[1, 2]), "unknown names"),
])
def test_calibration_schema_errors(tmp_path, edit, match):
    with pytest.raises(ValidationError, match=match):
        load_case(_case(tmp_path, edit))


def test_validate_backtest(tmp_path):
    def edit(r):
        for k in ("precip_forecast", "etp_forecast"):  # replay history instead of forecast files
            r["data"].pop(k)
        r["calibration"] = {"simulation": {"start": "2016-01-01", "end": "2016-02-29"},
                            "objective": {"start": "2016-02-01", "end": "2016-02-29"}, "horizon": 14}
        r["sub_basins"]["avermelha"].pop("calibration_bounds")
    main(["validate", str(_case(tmp_path, edit)), "-o", str(tmp_path / "val")])
    prev = pd.read_csv(tmp_path / "val/previsao.csv", sep=";", parse_dates=["data_caso"])
    assert (prev.data_caso.dt.dayofweek == 3).all() and (prev.data_caso >= "2016-02-01").all()
    assert set(prev.nome) == {"stoantjari", "avermelha"}
    v = pd.read_csv(tmp_path / "val/validacao.csv", sep=";")
    assert {"diaria", "semanal"} <= set(v.discretizacao)
