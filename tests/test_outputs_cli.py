import pandas as pd
import pytest

from conftest import DATA
from pysmap.cli import main


@pytest.mark.slow
def test_run_yaml_writes_csv_and_ons_files(tmp_path):
    main(["run", str(DATA / "case_pmur.yml"), "-o", str(tmp_path), "--ons-txt"])
    for f in ("previsao.csv", "otimizacao.csv", "funcao_objetivo.csv", "ajuste.csv", "stoantjari_ajuste.txt",
              "avermelha_ecmwf_1_previsao.txt"):
        assert (tmp_path / f).exists(), f
    prev = pd.read_csv(tmp_path / "previsao.csv", sep=";")
    assert list(prev.columns) == ["data_caso", "data_previsao", "cenario", "nome", "variavel", "valor"]
    q = prev.query("nome == 'avermelha' and cenario == 'ecmwf_1' and variavel == 'Qcalc'")
    lines = (tmp_path / "avermelha_ecmwf_1_previsao.txt").read_text().splitlines()
    assert lines[0] == "Data       Qcal"
    assert lines[1] == f"01/06/2017 {q.valor.iloc[0]:09.2f}"
    assert len(lines) == 43
    aj = (tmp_path / "stoantjari_ajuste.txt").read_text().splitlines()
    assert aj[0] == "Data       TU     EB        SUP       Qajustada"
    assert aj[1].startswith("01/05/2017 ") and len(aj[1].split()) == 5 and len(aj) == 32


def test_evaluate_writes_table(tmp_path):
    out = tmp_path / "validacao.csv"
    main(["evaluate", str(DATA / "arq_saida/previsao.csv"), str(DATA / "arq_saida/vazao_observada.csv"),
          "-o", str(out)])
    v = pd.read_csv(out, sep=";")
    assert set(v.discretizacao) == {"diaria", "semanal", "mensal"}


def test_ons_forecast_one_file_per_run(tmp_path):
    from pysmap.outputs import write_ons_forecast
    prev = pd.DataFrame({"data_caso": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-08"] * 2),
                         "data_previsao": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-09"]),
                         "cenario": "c", "nome": "A", "variavel": "Qcalc", "valor": [1.0, 2, 3, 4]})
    files = write_ons_forecast(prev, tmp_path)
    assert sorted(f.name for f in files) == ["A_c_20200101_previsao.txt", "A_c_20200108_previsao.txt"]
    assert (tmp_path / "A_c_20200108_previsao.txt").read_text().splitlines()[1] == "08/01/2020 000003.00"
    from pysmap.outputs import read_ons_forecast
    back = read_ons_forecast(tmp_path)
    prev["nome"] = "a"
    pd.testing.assert_frame_equal(back, prev)


def test_legacy_case_folder_cli(tmp_path):
    out = tmp_path / "out"
    main(["convert", str(DATA / "legacy" / "Arq_Entrada"), "-o", str(tmp_path / "case.yml")])
    assert (tmp_path / "case.yml").exists() and (tmp_path / "vazoes.csv").exists()
    with pytest.raises(SystemExit, match="oficial"):
        main(["run", str(DATA / "legacy" / "Arq_Entrada"), "--model", "pmur", "-o", str(out)])


def test_ons_adjustment_one_file_per_run(tmp_path):
    from pysmap.outputs import read_ons_adjustment, write_ons_adjustment
    runs = pd.to_datetime(["2020-01-03", "2020-01-10"])
    rows = [(r, r - pd.Timedelta(days=k), v, val) for r in runs for k in (2, 1)
            for v, val in (("Tu", 0.5), ("Qbase", 1.0), ("Qsup1", 1.0), ("Qsup2", 0.0), ("Qplan", 0.0), ("Qcalc", 2.0))]
    assim = pd.DataFrame(rows, columns=["data_caso", "data_assimilacao", "variavel", "valor"]).assign(nome="A")
    files = write_ons_adjustment(assim, tmp_path)
    assert sorted(f.name for f in files) == ["A_20200103_ajuste.txt", "A_20200110_ajuste.txt"]
    back = read_ons_adjustment(tmp_path)
    assert list(back["data_caso"].unique()) == list(runs) and set(back["nome"]) == {"a"} and len(back) == 4
    assert (back["Tu"] == 0.5).all() and (back["Qsup"] == 1.0).all()
