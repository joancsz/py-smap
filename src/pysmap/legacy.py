"""Reader for ONS's legacy SMAP/ONS deck (`Arq_Entrada`: CASO.txt, bat.conf and per-sub-basin .txt files),
the input of smapOnsR's `le_arq_entrada` / `executa_caso_oficial`. Runs with `forecast.OFICIAL`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import weighting
from .case import OBJECTIVE_CODES
from .forecast import Inputs
from .io import fill_forecast
from .params import KT_NAMES

PARAMS = ["Str", "K2t", "Crec", "Ai", "Capc", "K_kt", "K2t2", "H1", "H", "K3t", "K1t", "Ecof", "Pcof", "Ecof2"]
LIMITS = ["limite_superior_ebin", "limite_inferior_ebin", "limite_superior_prec", "limite_inferior_prec"]
DAY = pd.Timedelta(days=1)


class _Folder:
    """Case-insensitive file lookup: deck files mix upper and lower case freely."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.files = {f.name.lower(): f for f in self.path.iterdir()}

    def get(self, name: str, required: bool = True) -> Path | None:
        f = self.files.get(name.lower())
        if f is None and required:
            raise ValueError(f"{self.path}: missing file {name}")
        return f


def _lines(path: Path) -> list[str]:
    """Non-blank lines, each cut at the `'` that starts its comment."""
    out = [ln.split("'")[0].strip() for ln in path.read_text(encoding="latin-1").splitlines()]
    return [ln for ln in out if ln]


def _float(s: str, what: str) -> float:
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{what}: non-numeric value {s!r}") from None


def _count(lines: list[str], path: Path, what: str) -> list[str]:
    """First line declares how many entries follow; return them."""
    n = _float(lines[0].split()[0], f"{path} {what} count") if lines else 0
    if n <= 0 or n != int(n):
        raise ValueError(f"{path}: {what} count must be a positive integer")
    if len(lines) - 1 != n:
        raise ValueError(f"{path}: declares {int(n)} {what}, has {len(lines) - 1}")
    return lines[1:]


def read_caso(folder: _Folder) -> list[str]:
    path = folder.get("caso.txt")
    names = [ln.lower() for ln in _count(_lines(path), path, "sub-basins")]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicated sub-basin")
    return names


def read_bat_conf(folder: _Folder) -> dict:
    path = folder.get("bat.conf")
    conf = dict(ln.split("=", 1) for ln in path.read_text(encoding="latin-1").split() if "=" in ln)
    if conf.get("objetivo") not in ("dm", "nse", "mape"):
        raise ValueError(f"{path}: objetivo must be dm, nse or mape")
    conf["objetivo"] = OBJECTIVE_CODES[conf["objetivo"]]
    conf["arqPrev"] = int(conf.get("arqPrev", 0))
    conf["ajusta_precipitacao"] = int(conf.get("ajusta_precipitacao", 0))
    if conf["arqPrev"] not in (0, 1, 2) or conf["ajusta_precipitacao"] not in (0, 1):
        raise ValueError(f"{path}: arqPrev must be 0..2 and ajusta_precipitacao 0 or 1")
    return conf


def read_parametros(folder: _Folder, name: str) -> tuple[list[tuple], list[tuple]]:
    """`nome, parametro, valor` rows and the assimilation-bound initialization rows."""
    path = folder.get(f"{name}_parametros.txt")
    lines = _lines(path)
    if len(lines) < 20:
        raise ValueError(f"{path}: expected 20 lines, got {len(lines)}")
    kt_line = lines[1].split()
    n = int(_float(kt_line[0], f"{path} Kt count"))
    if n != len(kt_line) - 1 or not 0 < n <= len(KT_NAMES):
        raise ValueError(f"{path}: declares {n} Kt weights, has {len(kt_line) - 1}")
    kt = [_float(v, f"{path} Kt") for v in kt_line[1:]][::-1]  # listed oldest first; KT_NAMES starts at lag 2
    if not 0.995 <= sum(kt) <= 1.005:
        raise ValueError(f"{path}: Kt weights sum to {sum(kt):.4f}, expected 1")
    rows = [(name, "Area", _float(lines[0], f"{path} area"))] + list(zip([name] * n, KT_NAMES, kt))
    rows += [(name, p, _float(v, f"{path} {p}")) for p, v in zip(PARAMS, lines[2:16])]
    limits = [(name, v, _float(s, f"{path} {v}"), 0) for v, s in zip(LIMITS, lines[16:20])]
    if any(r[2] < 0 for r in rows + limits):
        raise ValueError(f"{path}: negative value")
    return rows, limits


def read_inicializacao(folder: _Folder, name: str) -> tuple[pd.Timestamp, int, list[tuple]]:
    """Run date, horizon and the Ebin/Supin/Tuin/numero_dias_assimilacao rows (Tuin as a fraction)."""
    path = folder.get(f"{name}_inicializacao.txt")
    lines = _lines(path)
    if len(lines) < 6:
        raise ValueError(f"{path}: expected 6 lines, got {len(lines)}")
    date = pd.to_datetime(lines[0], format="%d/%m/%Y", errors="coerce")
    if pd.isna(date):
        raise ValueError(f"{path}: bad run date {lines[0]!r}")
    warmup, horizon, ebin, supin, tuin = (_float(v, str(path)) for v in lines[1:6])
    if warmup < 2 or horizon <= 0 or min(ebin, supin, tuin) < 0 or tuin > 100:
        raise ValueError(f"{path}: needs warm-up days >= 2, horizon > 0, Ebin/Supin/Tuin >= 0 and Tuin <= 100")
    rows = [(name, "Ebin", ebin, 0), (name, "Supin", supin, 0), (name, "Tuin", tuin / 100, 0),
            (name, "numero_dias_assimilacao", warmup - 1, 0)]  # the file counts day d itself
    return date, int(horizon), rows


def read_evapotranspiracao(folder: _Folder, name: str) -> pd.DataFrame:
    path = folder.get(f"{name}_evapotranspiracao.txt")
    rows = [ln.split()[:2] for ln in _lines(path)]
    df = pd.DataFrame({"mes": [int(_float(m, f"{path} month")) for m, _ in rows], "nome": name,
                       "valor": [_float(v, f"{path} ETP") for _, v in rows]})
    if sorted(df["mes"]) != list(range(1, 13)) or (df["valor"] <= 0).any():
        raise ValueError(f"{path}: needs months 1..12 once each, with positive ETP")
    return df


def _no_gaps(dates: pd.Series, path: Path):
    full = pd.date_range(dates.min(), dates.max())
    if len(full) != len(dates) or dates.duplicated().any():
        missing = full.difference(dates)
        raise ValueError(f"{path}: missing or duplicated dates {list(missing.date[:5])}")


def read_vazao(folder: _Folder, name: str) -> pd.DataFrame:
    """`code|...|...|...|YYYY-MM-DD hh:mm:ss|flow`; '-' or blank marks a missing value."""
    path = folder.get(f"{name}.txt")
    df = pd.read_csv(path, sep="|", header=None, dtype=str, encoding="latin-1")
    if df.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 '|'-separated columns")
    df = df[[4, 5]].set_axis(["data", "valor"], axis=1)
    df["valor"] = df["valor"].str.strip()
    df = df[df["valor"].notna() & (df["valor"] != "-")]
    out = pd.DataFrame({"data": pd.to_datetime(df["data"].str[:10], format="%Y-%m-%d", errors="coerce"),
                        "nome": name, "valor": pd.to_numeric(df["valor"], errors="coerce")})
    if out[["data", "valor"]].isna().any().any() or (out["valor"] < 0).any():
        raise ValueError(f"{path}: bad date, non-numeric or negative flow")
    out = out.sort_values("data", ignore_index=True)
    _no_gaps(out["data"], path)
    return out


def read_postos_plu(folder: _Folder, name: str) -> pd.DataFrame:
    path = folder.get(f"{name}_postos_plu.txt")
    rows = [ln.split() for ln in _count(_lines(path), path, "rain gauges")]
    if any(len(r) != 2 for r in rows):
        raise ValueError(f"{path}: expected `gauge weight` lines")
    g = pd.DataFrame({"nome": name, "posto": [p.lower().removeprefix("0") for p, _ in rows],
                      "valor": [_float(w, f"{path} weight") for _, w in rows]})
    if (g["valor"] < 0).any() or not 0.995 <= g["valor"].sum() <= 1.005:
        raise ValueError(f"{path}: gauge weights must be >= 0 and sum to 1")
    return g


def read_chuva(folder: _Folder, posto: str) -> pd.DataFrame:
    """`<posto>_c.txt` (or `0<posto>_c.txt`): `gauge dd/mm/YYYY hhmm rain`."""
    path = folder.get(f"{posto}_c.txt", required=False) or folder.get(f"0{posto}_c.txt")
    rows = [ln.split() for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    if any(len(r) != 4 for r in rows):
        raise ValueError(f"{path}: expected 4 columns on every line")
    out = pd.DataFrame({"data": pd.to_datetime([r[1] for r in rows], format="%d/%m/%Y", errors="coerce"),
                        "posto": posto, "valor": pd.to_numeric([r[3] for r in rows], errors="coerce")})
    if out[["data", "valor"]].isna().any().any() or (out["valor"] < 0).any():
        raise ValueError(f"{path}: bad date, non-numeric or negative rain")
    out = out.sort_values("data", ignore_index=True)
    _no_gaps(out["data"], path)
    return out


def read_modelos(folder: _Folder) -> list[str]:
    path = folder.get("modelos_precipitacao.txt")
    names = [ln.lower() for ln in _count(_lines(path), path, "scenarios")]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicated scenario")
    return names


def read_pontos_grade(folder: _Folder, name: str, scenarios: list[str]) -> pd.DataFrame:
    """Grid points (lon, lat) of a sub-basin, from `<sb>_<model>.txt` where model is the scenario up to `_`."""
    parts = []
    for model in dict.fromkeys(s.split("_")[0] for s in scenarios):
        path = folder.get(f"{name}_{model}.txt")
        pts = [ln.split()[:2] for ln in _count(_lines(path), path, "grid points")]
        lon, lat = (np.array([_float(p[i], f"{path} coordinate") for p in pts]) for i in (0, 1))
        if (np.abs(lon) >= 100).any() or (np.abs(lat) >= 100).any():
            raise ValueError(f"{path}: |longitude| and |latitude| must be < 100")
        parts += [pd.DataFrame({"nome": name, "cenario": s, "lon": lon.round(4), "lat": lat.round(4)})
                  for s in scenarios if s.split("_")[0] == model]
    return pd.concat(parts, ignore_index=True)


def _dat(path: Path, first_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    names = first_cols + [str(i) for i in range(1, df.shape[1] - len(first_cols) + 1)]
    df.columns = names
    for c in names:
        if c != "cenario":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.drop(columns="cenario", errors="ignore").isna().any().any():
        raise ValueError(f"{path}: non-numeric value")
    df["lon"], df["lat"] = df["lon"].round(4), df["lat"].round(4)
    return df.melt(id_vars=first_cols, var_name="dia", value_name="valor").astype({"dia": int})


def read_previsao(folder: _Folder, kind: int, d: pd.Timestamp, horizon: int, scenarios: list[str]) -> pd.DataFrame:
    """Grid-point rain forecast as `cenario, lon, lat, dia (1 = d+1), valor`, in the bat.conf `arqPrev` format:
    0 one file per scenario and day, 1 one file per scenario, 2 one file for all scenarios."""
    tag = f"p{d:%d%m%y}"
    if kind == 2:
        return _dat(folder.get(f"precipitacao_prevista_{tag}.dat"), ["cenario", "lon", "lat"]).assign(
            cenario=lambda x: x["cenario"].str.lower())
    parts = []
    for s in scenarios:
        if kind == 1:
            parts.append(_dat(folder.get(f"{s}_{tag}a{d + horizon * DAY:%d%m%y}.dat"), ["lon", "lat"]).assign(cenario=s))
            continue
        files = [folder.get(f"{s}_{tag}a{d + i * DAY:%d%m%y}.dat", required=False) for i in range(1, horizon + 1)]
        last = max((i for i, f in enumerate(files) if f), default=-1)
        if last < 0 or not all(files[: last + 1]):
            raise ValueError(f"{folder.path}: {s} forecast files must start at d+1 with no missing day in between")
        for i, f in enumerate(files[: last + 1], start=1):
            parts.append(_dat(f, ["lon", "lat"]).assign(cenario=s, dia=i))
    return pd.concat(parts, ignore_index=True)[["cenario", "lon", "lat", "dia", "valor"]]


def read_legacy_case(folder: str | Path) -> Inputs:
    """Read an ONS `Arq_Entrada` folder (R's le_arq_entrada). Single run date, as in the official deck."""
    folder = _Folder(folder)
    names = read_caso(folder)
    conf = read_bat_conf(folder)
    scenarios = read_modelos(folder)

    params, init, etp, flow, gauges, dates, points = [], [], [], [], [], set(), []
    for name in names:
        rows, limits = read_parametros(folder, name)
        d, horizon, init_rows = read_inicializacao(folder, name)
        params += rows
        init += init_rows + limits + [(name, "funcao_objetivo", conf["objetivo"], 0),
                                      (name, "ajusta_precipitacao", conf["ajusta_precipitacao"], 0)]
        dates.add((d, horizon))
        etp.append(read_evapotranspiracao(folder, name))
        flow.append(read_vazao(folder, name))
        gauges.append(read_postos_plu(folder, name))
        points.append(read_pontos_grade(folder, name, scenarios))
    if len(dates) != 1:
        raise ValueError(f"{folder.path}: sub-basins disagree on run date or horizon: {sorted(dates)}")
    (d, horizon), = dates
    run_dates = pd.DataFrame({"data": [d], "numero_dias_previsao": [horizon]})
    gauges = pd.concat(gauges, ignore_index=True)

    rain = pd.concat([read_chuva(folder, p) for p in gauges["posto"].unique()], ignore_index=True)
    fc = read_previsao(folder, conf["arqPrev"], d, horizon, scenarios)
    fc = pd.concat(points).merge(fc, on=["cenario", "lon", "lat"])  # grid points outside the sub-basins drop out
    missing = set(names) - set(fc["nome"])
    if missing:
        raise ValueError(f"{folder.path}: no forecast grid point matches sub-basins {sorted(missing)}")
    fc = fc.assign(data_rodada=d, data_previsao=d + fc["dia"] * DAY).groupby(
        ["data_rodada", "data_previsao", "cenario", "nome"], as_index=False)["valor"].mean()
    if (fc["valor"] < 0).any():
        raise ValueError(f"{folder.path}: negative rain forecast")

    return Inputs(
        parameters=pd.DataFrame(params, columns=["nome", "parametro", "valor"]),
        initialization=pd.DataFrame(init, columns=["nome", "variavel", "valor", "mes"]),
        precip_observed=weighting.spatial(rain, gauges),
        precip_forecast=fill_forecast(fc, run_dates, extra_days=2),
        flow_observed=pd.concat(flow, ignore_index=True),
        run_dates=run_dates, sub_basins=names,
        etp_climatology=pd.concat(etp, ignore_index=True))
