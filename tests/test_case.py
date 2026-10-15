import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from conftest import DATA
from pysmap import forecast
from pysmap.case import load_case
from pysmap.forecast import _init_value
from pysmap.io import read_case

BASINS = ["stoantjari", "avermelha"]


@pytest.mark.slow
def test_yaml_matches_csv_folder():
    case = load_case(DATA / "case_pmur.yml")
    assert case.config == forecast.PMUR_ETP
    csv = read_case(DATA / "arq_entrada_pmur")
    csv.sub_basins = BASINS
    a, b = forecast.run_chained(case.inputs, case.config), forecast.run_chained(csv, forecast.PMUR_ETP)
    pd.testing.assert_frame_equal(a["previsao"], b["previsao"])
    pd.testing.assert_frame_equal(a["funcao_objetivo"], b["funcao_objetivo"])


def _write(tmp_path, edit):
    raw = yaml.safe_load((DATA / "case_pmur.yml").read_text())
    raw["data"] = {k: str(DATA / v) for k, v in raw["data"].items()}
    edit(raw)
    p = tmp_path / "case.yml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("edit, field", [
    (lambda r: r["sub_basins"]["avermelha"]["kt"].update({0: 0.0}), "kt"),
    (lambda r: r["sub_basins"]["avermelha"]["params"].pop("pmur"), "pmur"),
    (lambda r: r["model"].update(etp="climatology", assimilate_etp=False), "etp_climatology"),
    (lambda r: r["sub_basins"]["avermelha"].update(colour="blue"), "colour"),
    (lambda r: r["sub_basins"]["avermelha"]["initial"].update(tuin=150), "tuin"),
])
def test_invalid_case_names_the_field(tmp_path, edit, field):
    with pytest.raises(ValidationError, match=field):
        load_case(_write(tmp_path, edit))


def test_defaults_merge_and_monthly_bounds(tmp_path):
    def edit(r):
        r["defaults"]["assimilation"]["bounds"]["prec"] = {m: [0.5, 1.5 if m == 6 else 2] for m in range(1, 13)}
        r["sub_basins"]["avermelha"]["assimilation"] = {"days": 20}
    init = load_case(_write(tmp_path, edit)).inputs.initialization
    assert _init_value(init, "avermelha", "numero_dias_assimilacao") == 20
    assert _init_value(init, "stoantjari", "numero_dias_assimilacao") == 31
    assert _init_value(init, "avermelha", "limite_superior_prec", 6) == 1.5
    assert _init_value(init, "avermelha", "limite_superior_prec", 7) == 2
    assert _init_value(init, "avermelha", "limite_superior_ebin", 6) == 1.2


def test_example_case_is_valid():
    case = load_case(DATA.parent.parent / "examples" / "case.yml")
    assert case.inputs.sub_basins == ["stoantjari"]
