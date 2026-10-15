"""Write run results: `;` CSVs (as executa_caso_oficial_v2) and ONS fixed-width text files; read the latter back."""
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
    """`<nome>_ajuste.txt`: soil moisture (%), base flow, surface flow and fitted flow per assimilation day.

    With several run dates, one file per run: `<nome>_<YYYYMMDD>_ajuste.txt`.
    """
    wide = assimilacao.pivot(index=["nome", "data_caso", "data_assimilacao"], columns="variavel",
                             values="valor").sort_index()
    many = assimilacao["data_caso"].nunique() > 1
    files = []
    for (name, run), g in wide.groupby(level=["nome", "data_caso"], sort=False):
        lines = ["Data       TU     EB        SUP       Qajustada"]
        for (_, _, d), r in g.iterrows():
            sup = r["Qsup1"] + r["Qsup2"] + r["Qplan"]
            lines.append(f"{d:%d/%m/%Y} {_fmt(6, 2)(r['Tu'] * 100)} {_fmt(9, 2)(r['Qbase'])} {_fmt(9, 2)(sup)} "
                         f"{_fmt(9, 2)(r['Qcalc'])}")
        files.append(folder / (f"{name}_{run:%Y%m%d}_ajuste.txt" if many else f"{name}_ajuste.txt"))
        files[-1].write_text("\n".join(lines) + "\n")
    return files


def _ons_txt(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=columns)
    df[columns[0]] = pd.to_datetime(df[columns[0]], format="%d/%m/%Y")
    return df


def read_ons_forecast(folder: str | Path, names: list[str] | None = None) -> pd.DataFrame:
    """Read `<nome>_<cenario>[_<YYYYMMDD>]_previsao.txt` files back as `previsao` rows (R's le_previsao).

    The run date is the file's date suffix, else its first date. Sub-basin names are lowercased; pass `names`
    when a name contains `_` (p_da_beira), otherwise the name is taken up to the first `_`.
    """
    parts = []
    for f in sorted(Path(folder).iterdir()):
        stem = f.name.lower()
        if not stem.endswith("_previsao.txt"):
            continue
        stem = stem.removesuffix("_previsao.txt")
        name = next((n for n in names or [] if stem.startswith(n.lower() + "_")), stem.split("_")[0]).lower()
        scen = stem[len(name) + 1:]
        run = scen[-8:] if scen[-9:-8] == "_" and scen[-8:].isdigit() else None
        scen = scen[:-9] if run else scen
        df = _ons_txt(f, ["data_previsao", "valor"])
        parts.append(df.assign(data_caso=pd.Timestamp(run) if run else df["data_previsao"].iloc[0],
                               cenario=scen, nome=name, variavel="Qcalc"))
    if not parts:
        raise ValueError(f"{folder}: no *_previsao.txt files")
    return pd.concat(parts, ignore_index=True)[["data_caso", "data_previsao", "cenario", "nome", "variavel", "valor"]]


def read_ons_adjustment(folder: str | Path) -> pd.DataFrame:
    """Read `<nome>[_<YYYYMMDD>]_ajuste.txt` files back (R's le_ajuste): nome, data_caso, data_assimilacao,
    Tu (fraction), Qbase, Qsup, Qcalc. The run date is the file's date suffix, else the day after its last row."""
    parts = []
    for f in sorted(Path(folder).iterdir()):
        stem = f.name.lower()
        if not stem.endswith("_ajuste.txt"):
            continue
        stem = stem.removesuffix("_ajuste.txt")
        run = stem[-8:] if stem[-9:-8] == "_" and stem[-8:].isdigit() else None
        df = _ons_txt(f, ["data_assimilacao", "Tu", "Qbase", "Qsup", "Qcalc"])
        parts.append(df.assign(nome=stem[:-9] if run else stem, data_caso=pd.Timestamp(run) if run
                               else df["data_assimilacao"].iloc[-1] + pd.Timedelta(days=1)))
    if not parts:
        raise ValueError(f"{folder}: no *_ajuste.txt files")
    df = pd.concat(parts, ignore_index=True)
    df["Tu"] /= 100
    return df[["nome", "data_caso", "data_assimilacao", "Tu", "Qbase", "Qsup", "Qcalc"]]
