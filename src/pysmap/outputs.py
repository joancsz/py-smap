"""Write run results: `;` CSVs (as executa_caso_oficial_v2) and ONS fixed-width text files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = {"previsao": "previsao.csv", "otimizacao": "otimizacao.csv", "funcao_objetivo": "funcao_objetivo.csv",
          "assimilacao": "ajuste.csv"}


def write_outputs(out: dict[str, pd.DataFrame], folder: str | Path, ons_txt: bool = False,
                  append: bool = False) -> list[Path]:
    """Write `run_chained` results to `folder`; returns the files written.

    append=True adds rows to existing CSVs, so results can be written one sub-basin at a time.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for key, fname in TABLES.items():
        out[key].to_csv(folder / fname, sep=";", index=False, date_format="%Y-%m-%d", mode="a" if append else "w",
                        header=not append)
        written.append(folder / fname)
    if ons_txt:
        written += write_ons_forecast(out["previsao"], folder) + write_ons_adjustment(out["assimilacao"], folder)
    return written


def _fmt(width: int, decimals: int):
    return lambda v: f"{v:0{width}.{decimals}f}"


def write_ons_forecast(previsao: pd.DataFrame, folder: Path) -> list[Path]:
    """`<nome>_<cenario>_previsao.txt`: header, then `dd/mm/YYYY Qcalc` (R's escreve_previsao).

    With several run dates, one file per run: `<nome>_<cenario>_<YYYYMMDD>_previsao.txt`.
    """
    q = previsao[previsao["variavel"] == "Qcalc"]
    many = q["data_caso"].nunique() > 1
    files = []
    for (name, scen, run), g in q.groupby(["nome", "cenario", "data_caso"], sort=False):
        lines = ["Data       Qcal"] + [f"{d:%d/%m/%Y} {_fmt(9, 2)(v)}" for d, v in zip(g["data_previsao"], g["valor"])]
        files.append(folder / (f"{name}_{scen}_{run:%Y%m%d}_previsao.txt" if many else f"{name}_{scen}_previsao.txt"))
        files[-1].write_text("\n".join(lines) + "\n")
    return files


def write_ons_adjustment(assimilacao: pd.DataFrame, folder: Path) -> list[Path]:
    """`<nome>_ajuste.txt`: soil moisture (%), base flow, surface flow and fitted flow per assimilation day."""
    wide = assimilacao.pivot(index=["nome", "data_caso", "data_assimilacao"], columns="variavel",
                             values="valor").sort_index()
    files = []
    for name, g in wide.groupby(level="nome", sort=False):
        lines = ["Data       TU     EB        SUP       Qajustada"]
        for (_, _, d), r in g.iterrows():
            sup = r["Qsup1"] + r["Qsup2"] + r["Qplan"]
            lines.append(f"{d:%d/%m/%Y} {_fmt(6, 2)(r['Tu'] * 100)} {_fmt(9, 2)(r['Qbase'])} {_fmt(9, 2)(sup)} "
                         f"{_fmt(9, 2)(r['Qcalc'])}")
        files.append(folder / f"{name}_ajuste.txt")
        files[-1].write_text("\n".join(lines) + "\n")
    return files
