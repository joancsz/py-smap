import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import DATA
from pysmap import forecast
from pysmap.case import load_case, to_yaml
from pysmap.forecast import _init_value
from pysmap.io import read_case
from pysmap.legacy import read_legacy_case
from pysmap.outputs import read_ons_adjustment, read_ons_forecast

DECK = DATA / "legacy" / "Arq_Entrada"


@pytest.fixture(scope="module")
def deck():
    return read_legacy_case(DECK)


def test_reads_deck(deck):
    assert len(deck.sub_basins) == 22 and deck.sub_basins[0] == "p_da_beira"
    p = deck.parameters.query("nome == 'p_da_beira'").set_index("parametro")["valor"]
    assert p["Area"] == 244841 and p["Str"] == 19.14 and p["Ecof2"] == 0.35
    assert p["Kt2"] == 0.001 and p["Kt1"] == 0.007 and p["Kt-60"] == 0.021  # listed oldest first
    init = deck.initialization
    assert _init_value(init, "p_da_beira", "Tuin") == pytest.approx(0.8452)
    assert _init_value(init, "p_da_beira", "numero_dias_assimilacao") == 31
    assert _init_value(init, "p_da_beira", "limite_superior_prec", 3) == 2.0
    assert deck.run_dates.iloc[0].tolist() == [pd.Timestamp("2023-02-23"), 19]
    fc = deck.precip_forecast.query("nome == 'p_da_beira'")
    assert fc["data_previsao"].min() == pd.Timestamp("2023-02-24") and len(fc) == 20  # d+1 .. d+n+1


def _official_forecast():
    return read_ons_forecast(DATA / "legacy" / "Arq_Saida", names=["p_da_beira"])


def test_matches_official_output(deck):
    out = forecast.run_chained(replace(deck, sub_basins=["p_da_beira"]), forecast.OFICIAL)
    q = out["previsao"].query("variavel == 'Qcalc'")["valor"].to_numpy()
    ref = _official_forecast()["valor"].to_numpy()
    assert len(q) == len(ref) == 19
    np.testing.assert_allclose(q, ref, rtol=0.01)  # R's test: day 10 = 3493 within 1%
    assert q[9] == pytest.approx(3491.56, rel=0.001)


def test_read_back_official_files():
    fc = _official_forecast()
    assert set(fc["nome"]) == {"p_da_beira"} and set(fc["cenario"]) == {"pmedia_orig"}
    assert fc["data_caso"].iloc[0] == pd.Timestamp("2023-02-23") and fc["valor"].iloc[9] == 3491.56
    aj = read_ons_adjustment(DATA / "legacy" / "Arq_Saida")
    last = aj.query("nome == 'p_da_beira'").iloc[-1]
    assert last["data_assimilacao"] == pd.Timestamp("2023-02-22") and last["Tu"] == pytest.approx(0.8978)
    assert last["Qcalc"] == 2301.45


def _broken(tmp_path, name, text):
    shutil.copytree(DECK, tmp_path / "d")
    if text is None:
        (tmp_path / "d" / name).unlink()
    else:
        (tmp_path / "d" / name).write_text(text)
    return tmp_path / "d"


@pytest.mark.parametrize("name, text, message", [
    ("CASO.txt", "3\nP_da_Beira\nJirau2\n", "declares 3 sub-basins"),
    ("P_da_Beira_PARAMETROS.txt", (DECK / "P_da_Beira_PARAMETROS.txt").read_text().replace("0.021", "0.5", 1),
     "Kt weights sum"),
    ("P_da_Beira_EVAPOTRANSPIRACAO.txt", None, "missing file p_da_beira_evapotranspiracao.txt"),
    ("bat.conf", "objetivo=kge\n", "objetivo must be"),
])
def test_bad_deck_names_the_problem(tmp_path, name, text, message):
    with pytest.raises(ValueError, match=message):
        read_legacy_case(_broken(tmp_path, name, text))


def test_convert_deck_to_yaml(tmp_path, deck):
    one = replace(deck, sub_basins=["p_da_beira"])
    case = load_case(to_yaml(one, forecast.OFICIAL, tmp_path / "case.yml"))
    assert case.config == forecast.OFICIAL
    a = forecast.run_chained(one, forecast.OFICIAL)["previsao"]
    b = forecast.run_chained(case.inputs, case.config)["previsao"]
    pd.testing.assert_frame_equal(a, b)


def test_convert_csv_folder_to_yaml(tmp_path):
    inputs = read_case(DATA / "arq_entrada_pmur")
    case = load_case(to_yaml(inputs, forecast.PMUR_ETP, tmp_path / "case.yml"))
    assert case.inputs.sub_basins == inputs.sub_basins and case.config == forecast.PMUR_ETP
    sb = case.spec.sub_basins["stoantjari"]
    assert sb.params.pmur is not None and sb.assimilation.days == 32 - 1  # the CSV counts day d itself


def test_forecast_file_layouts_agree(tmp_path, deck):
    """bat.conf arqPrev=1 (one file per scenario) and 2 (one file for all) give the same rain as arqPrev=0."""
    rows = [pd.read_csv(f, sep=r"\s+", header=None, names=["lon", "lat", "v"]).assign(dia=i)
            for i, f in enumerate(sorted(DECK.glob("PMEDIA_ORIG_p230223a*.dat"),
                                         key=lambda f: pd.to_datetime(f.stem[-6:], format="%d%m%y")), start=1)]
    wide = pd.concat(rows).pivot_table(index=["lon", "lat"], columns="dia", values="v").reset_index()
    for kind, name, table in ((1, "PMEDIA_ORIG_p230223a140323.dat", wide),
                              (2, "precipitacao_prevista_p230223.dat", wide.assign(c="PMEDIA_ORIG")[
                                  ["c", *wide.columns]])):
        d = tmp_path / str(kind)
        shutil.copytree(DECK, d, ignore=shutil.ignore_patterns("PMEDIA_ORIG_p*.dat"))
        (d / "bat.conf").write_text((DECK / "bat.conf").read_text() + f"\narqPrev={kind}\n")
        table.to_csv(d / name, sep=" ", header=False, index=False)
        pd.testing.assert_frame_equal(read_legacy_case(d).precip_forecast, deck.precip_forecast)
