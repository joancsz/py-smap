"""Reader for the smapOnsR "new format" input folder (arquivos.csv + semicolon CSVs)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import weighting
from .forecast import Inputs

KINDS = {"SUB_BACIAS", "PARAMETROS", "VAZAO_OBSERVADA", "EVAPOTRANSPIRACAO_NC", "EVAPOTRANSPIRACAO_OBSERVADA",
         "POSTOS_PLUVIOMETRICOS", "PRECIPITACAO_OBSERVADA", "INICIALIZACAO", "DATAS_RODADAS",
         "PRECIPITACAO_PREVISTA", "EVAPOTRANSPIRACAO_PREVISTA"}
INIT_VARS = {"Ebin", "Supin", "Tuin", "numero_dias_assimilacao", "limite_inferior_ebin", "limite_superior_ebin",
             "limite_superior_prec", "limite_inferior_prec", "funcao_objetivo", "ajusta_precipitacao"}


def _csv(path: Path) -> pd.DataFrame:
    if Path(path).suffix == ".parquet":  # needs pyarrow installed
        df = pd.read_parquet(path).astype(str)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()
    return df.apply(lambda c: c.str.strip())


def _num(s: pd.Series, what: str) -> pd.Series:
    out = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")  # some files use decimal commas
    if out.isna().any():
        raise ValueError(f"{what}: non-numeric value {s[out.isna()].iloc[0]!r}")
    if (out < 0).any():
        raise ValueError(f"{what}: negative value {out[out < 0].iloc[0]}")
    return out


def _date(s: pd.Series, what: str) -> pd.Series:
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.to_datetime(s.where(~iso), format="%d/%m/%Y", errors="coerce")
    out[iso] = pd.to_datetime(s[iso], format="ISO8601")
    if out.isna().any():
        raise ValueError(f"{what}: bad date {s[out.isna()].iloc[0]!r}")
    return out


def _wide_history(path: Path, names=None) -> pd.DataFrame:
    """`data;<name1>;<name2>...` table -> long `data, nome, valor` for the requested names (default: all)."""
    df = _csv(path)
    if df.columns[0] != "data":
        raise ValueError(f"{path}: first column must be 'data'")
    df["data"] = _date(df["data"], str(path))
    if df["data"].duplicated().any():
        raise ValueError(f"{path}: duplicated date {df['data'][df['data'].duplicated()].iloc[0].date()}")
    names = list(df.columns[1:]) if names is None else names
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    long = df[["data", *names]].melt(id_vars="data", var_name="nome", value_name="valor")
    long["valor"] = _num(long["valor"], str(path))
    return long.sort_values(["nome", "data"], ignore_index=True)


def _forecast(path: Path) -> pd.DataFrame:
    df = _csv(path)
    if list(df.columns) != ["data_rodada", "data_previsao", "cenario", "nome", "valor"]:
        raise ValueError(f"{path}: columns must be data_rodada;data_previsao;cenario;nome;valor")
    for c in ("data_rodada", "data_previsao"):
        df[c] = _date(df[c], f"{path} {c}")
    df["valor"] = _num(df["valor"], str(path))
    if df.duplicated(["data_rodada", "data_previsao", "cenario", "nome"]).any():
        raise ValueError(f"{path}: duplicated forecast rows")
    return df


def fill_forecast(fc: pd.DataFrame, run_dates: pd.DataFrame, extra_days: int) -> pd.DataFrame:
    """Fill missing days d+1 .. d+n-1+extra_days of each (run, sub-basin, scenario) with its mean (R's completa_previsao).

    shortcut: R joins the mean on (nome, cenario) only, which mixes run dates when there are several;
    this uses each run's own mean.
    """
    keys = ["data_rodada", "nome", "cenario"]
    days = pd.concat([pd.DataFrame({"data_rodada": d, "data_previsao": pd.date_range(
        d + pd.Timedelta(days=1), d + pd.Timedelta(days=int(n) - 1 + extra_days))})
        for d, n in zip(run_dates["data"], run_dates["numero_dias_previsao"])])
    expected = fc[keys].drop_duplicates().merge(days, on="data_rodada")
    m = expected.merge(fc[[*keys, "data_previsao"]], how="left", indicator=True)
    missing = m[m["_merge"] == "left_only"].drop(columns="_merge")
    missing = missing.merge(fc.groupby(keys, as_index=False)["valor"].mean(), on=keys)
    return pd.concat([fc, missing[fc.columns]], ignore_index=True).sort_values(
        ["nome", "data_rodada", "data_previsao", "cenario"], ignore_index=True)


def history_as_forecast(obs: pd.DataFrame, run_dates: pd.DataFrame, first_day: int, last_extra: int) -> pd.DataFrame:
    """Replay observed days d+first_day .. d+n+last_extra as a single 'historico' scenario."""
    parts = []
    for d, n in zip(run_dates["data"], run_dates["numero_dias_previsao"]):
        sel = obs[(obs["data"] >= d + pd.Timedelta(days=first_day))
                  & (obs["data"] <= d + pd.Timedelta(days=int(n) + last_extra))]
        parts.append(sel.rename(columns={"data": "data_previsao"}).assign(data_rodada=d, cenario="historico"))
    return pd.concat(parts, ignore_index=True)[["data_rodada", "data_previsao", "cenario", "nome", "valor"]]


def read_case(folder: str | Path) -> Inputs:
    """Read a new-format case folder (see smapOnsR's inst/extdata/arq_entrada_novo)."""
    folder = Path(folder)
    index = _csv(folder / "arquivos.csv")
    unknown = set(index["arquivo"]) - KINDS
    if unknown:
        raise ValueError(f"arquivos.csv: unknown file kinds {sorted(unknown)}")
    files = {k: folder / f for k, f in zip(index["arquivo"], index["nome_arquivo"])}
    if "EVAPOTRANSPIRACAO_NC" in files and ({"EVAPOTRANSPIRACAO_OBSERVADA", "EVAPOTRANSPIRACAO_PREVISTA"} & set(files)):
        raise ValueError("arquivos.csv: give ETP either as climatology (NC) or as observed/forecast series")
    for k in ("SUB_BACIAS", "PARAMETROS", "INICIALIZACAO", "DATAS_RODADAS", "VAZAO_OBSERVADA",
              "POSTOS_PLUVIOMETRICOS", "PRECIPITACAO_OBSERVADA"):
        if k not in files:
            raise ValueError(f"arquivos.csv: missing {k}")

    basins = list(_csv(files["SUB_BACIAS"])["nome"])

    params = _csv(files["PARAMETROS"])
    params = params[~params["parametro"].str.contains("data_|_periodo")].copy()
    params["valor"] = _num(params["valor"], "parametros")
    if params.duplicated(["nome", "parametro"]).any():
        raise ValueError("parametros: duplicated parameter")
    kt_sum = params[params["parametro"].str.match(r"^Kt-?\d+$")].groupby("nome")["valor"].sum()
    bad = kt_sum[(kt_sum > 1.005) | (kt_sum < 0.995)]
    if len(bad):
        raise ValueError(f"parametros: Kt weights of {bad.index[0]} sum to {bad.iloc[0]}")

    init = _csv(files["INICIALIZACAO"])
    init["valor"] = _num(init["valor"], "inicializacao")
    init["mes"] = pd.to_numeric(init["mes"]) if "mes" in init else 0
    if not set(init["variavel"]) <= INIT_VARS:
        raise ValueError(f"inicializacao: unknown variables {sorted(set(init['variavel']) - INIT_VARS)}")
    if (init.query("variavel == 'Tuin'")["valor"] > 100).any():
        raise ValueError("inicializacao: Tuin is a percentage and must be <= 100")
    if (init.query("variavel == 'numero_dias_assimilacao'")["valor"] < 2).any():
        raise ValueError("inicializacao: numero_dias_assimilacao must be >= 2")
    init.loc[init["variavel"] == "Tuin", "valor"] /= 100
    init.loc[init["variavel"] == "numero_dias_assimilacao", "valor"] -= 1  # the file counts day d itself

    dates = _csv(files["DATAS_RODADAS"])
    dates["data"] = _date(dates["data"], "datasRodadas")
    dates["numero_dias_previsao"] = _num(dates["numero_dias_previsao"], "datasRodadas").astype(int)
    if (dates["numero_dias_previsao"] == 0).any() or dates["data"].duplicated().any():
        raise ValueError("datasRodadas: horizons must be > 0 and dates unique")
    dates = dates.sort_values("data", ignore_index=True)

    gauges = _csv(files["POSTOS_PLUVIOMETRICOS"])
    gauges = gauges[gauges["nome"].isin(basins)].copy()
    gauges["valor"] = _num(gauges["valor"], "postos_plu")
    g_sum = gauges.groupby("nome")["valor"].sum()
    if ((g_sum > 1.005) | (g_sum < 0.995)).any():
        raise ValueError("postos_plu: gauge weights of a sub-basin must sum to 1")
    for what, df in (("parametros", params), ("inicializacao", init), ("postos_plu", gauges)):
        missing = set(basins) - set(df["nome"])
        if missing:
            raise ValueError(f"{what}: missing sub-basins {sorted(missing)}")

    etp_nc = None
    if "EVAPOTRANSPIRACAO_NC" in files:
        etp_nc = _csv(files["EVAPOTRANSPIRACAO_NC"])
        etp_nc["valor"] = _num(etp_nc["valor"], "evapotranspiracao_nc")
        etp_nc["mes"] = _num(etp_nc["mes"], "evapotranspiracao_nc").astype(int)
    series = _load_series(files, basins, gauges, dates)
    return Inputs(parameters=params, initialization=init, run_dates=dates, sub_basins=basins,
                  etp_climatology=etp_nc, **series)


def _load_series(files: dict, basins: list, gauges: pd.DataFrame, dates: pd.DataFrame) -> dict:
    """Observed and forecast series for `Inputs`, keyed by arquivos.csv kinds. Shared with the YAML loader."""
    flow = _wide_history(files["VAZAO_OBSERVADA"], basins)
    rain_gauges = _wide_history(files["PRECIPITACAO_OBSERVADA"], list(gauges["posto"].unique()))
    rain = weighting.spatial(rain_gauges.rename(columns={"nome": "posto"}), gauges)

    if "PRECIPITACAO_PREVISTA" in files:
        fc = _forecast(files["PRECIPITACAO_PREVISTA"]).rename(columns={"nome": "posto"})
        missing = set(gauges["posto"]) - set(fc["posto"])
        if missing:
            raise ValueError(f"precipitation forecast: missing gauges {sorted(missing)[:5]}")
        rain_fc = fill_forecast(weighting.spatial(fc, gauges, by=("data_rodada", "data_previsao", "cenario")),
                                dates, extra_days=2)
    else:
        rain_fc = history_as_forecast(rain, dates, first_day=1, last_extra=2)

    etp_obs = etp_fc = None
    if "EVAPOTRANSPIRACAO_OBSERVADA" in files:
        etp_obs = _wide_history(files["EVAPOTRANSPIRACAO_OBSERVADA"], basins)
        if "EVAPOTRANSPIRACAO_PREVISTA" in files:
            etp_fc = fill_forecast(_forecast(files["EVAPOTRANSPIRACAO_PREVISTA"]), dates, extra_days=0)
        else:
            etp_fc = history_as_forecast(etp_obs, dates, first_day=0, last_extra=0)
    return dict(precip_observed=rain, precip_forecast=rain_fc, flow_observed=flow, etp_observed=etp_obs,
                etp_forecast=etp_fc)
