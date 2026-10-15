"""YAML case files: one human-readable file for model settings, parameters and initial state.

Time series stay in CSV/Parquet files that the YAML points to. See examples/case.yml.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator, model_validator

from .forecast import Inputs, RunConfig, _init_value
from .io import _load_series
from .params import KT_NAMES

Bounds = tuple[float, float]
OBJECTIVE_CODES = {"dm": 0, "nse": 1, "mape": 2, "kge": 3}
# YAML name -> R parameter name, as SmapParams.from_table reads it
PARAM_NAMES = {"str_": "Str", "k2t": "K2t", "crec": "Crec", "capc": "Capc", "k_kt": "K_kt", "h1": "H1",
               "k2t2": "K2t2", "h": "H", "k1t": "K1t", "k3t": "K3t", "pcof": "Pcof", "ecof": "Ecof",
               "ecof2": "Ecof2", "ai": "Ai", "lambda_": "Lambda", "pmur": "Pmur"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ModelCfg(_Strict):
    runoff: Literal["ai", "pmur"] = "ai"
    etp: Literal["climatology", "series"] = "climatology"
    assimilate: bool = True
    assimilate_etp: bool = False


class RunDate(_Strict):
    date: dt.date
    horizon: PositiveInt


class RunRange(_Strict):
    start: dt.date
    end: dt.date
    every: PositiveInt = 7
    horizon: PositiveInt


class DataPaths(_Strict):
    precip_observed: Path  # wide: data;<gauge>;...
    flow_observed: Path  # wide: data;<sub-basin>;...
    precip_forecast: Path | None = None  # data_rodada;data_previsao;cenario;nome(gauge);valor
    etp_observed: Path | None = None
    etp_forecast: Path | None = None


class Initial(_Strict):
    ebin: float = Field(ge=0)
    supin: float = Field(ge=0)
    tuin: float = Field(ge=0, le=100, description="soil moisture, % of Str")


class Assimilation(_Strict):
    days: int = Field(ge=2, description="assimilation window, days before the forecast date")
    objective: Literal["dm", "nse", "mape", "kge"] = "dm"
    adjust_last_day: bool = False
    bounds: dict[Literal["prec", "ebin"], Union[Bounds, dict[int, Bounds]]]

    @field_validator("bounds")
    @classmethod
    def _monthly_complete(cls, v):
        for k, b in v.items():
            if isinstance(b, dict) and set(b) != set(range(1, 13)):
                raise ValueError(f"monthly bounds for {k} need months 1..12")
        for k in ("prec", "ebin"):
            if k not in v:
                raise ValueError(f"missing bounds for {k}")
        return v


class Params(_Strict):
    str_: PositiveFloat = Field(alias="str")
    k2t: PositiveFloat
    crec: float = Field(ge=0)
    capc: float = Field(ge=0)
    k_kt: PositiveFloat
    h1: float = Field(ge=0)
    k2t2: PositiveFloat
    h: float = Field(ge=0)
    k1t: PositiveFloat
    k3t: PositiveFloat
    pcof: float = Field(ge=0)
    ecof: float = Field(ge=0)
    ecof2: float = Field(ge=0)
    ai: float | None = Field(None, ge=0)
    lambda_: float | None = Field(None, alias="lambda", ge=0)
    pmur: float | None = Field(None, ge=0)


class Period(_Strict):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _ordered(self):
        if self.start > self.end:
            raise ValueError("start is after end")
        return self


class CalibrationCfg(_Strict):
    simulation: Period
    objective: Period | None = None  # days scored; default: the simulation period
    exclude: list[tuple[dt.date, dt.date]] = []  # days weighted 0 (R's periodo desconsiderado)
    metric: Literal["dm", "nse", "mape", "kge"] = "dm"
    step: PositiveFloat = 1e-3  # finite-difference step (R's ndeps)
    scale_by_upper: bool = False  # optimize in units of the upper bound (R's tipo_escala = 1)
    maxiter: PositiveInt = 100
    horizon: PositiveInt = 42  # backtest horizon; run dates weekly, monthly when > 42

    @model_validator(mode="after")
    def _inside(self):
        o = self.objective
        if o and (o.start < self.simulation.start or o.end > self.simulation.end):
            raise ValueError("objective period must lie inside the simulation period")
        return self


class Relative(_Strict):
    times: Bounds  # bounds as multiples of the current value


class KtShape(_Strict):
    alfa: PositiveFloat
    ex: float = Field(gt=0, lt=1, description="R's Beta parameter of cria_kt")


class SubBasin(_Strict):
    area: PositiveFloat
    params: Params
    kt: dict[int, float] = Field(description="lag -> weight; 2, 1 = future days, 0 = today, -1..-60 = past")
    gauges: dict[str, float]
    initial: Initial
    assimilation: Assimilation
    etp_climatology: list[float] | None = Field(None, min_length=12, max_length=12)
    kt_shape: KtShape | None = None  # when calibrating, Kt is generated from this Beta shape
    calibration_bounds: dict[str, Union[Bounds, Relative]] = {}

    @field_validator("kt")
    @classmethod
    def _kt(cls, v):
        if not set(v) <= set(range(-60, 3)):
            raise ValueError("kt lags must be between -60 and 2")
        if not 0.995 <= sum(v.values()) <= 1.005:
            raise ValueError(f"kt weights sum to {sum(v.values()):.4f}, expected 1")
        return v

    @field_validator("gauges")
    @classmethod
    def _gauges(cls, v):
        if not v or not 0.995 <= sum(v.values()) <= 1.005:
            raise ValueError(f"gauge weights sum to {sum(v.values()):.4f}, expected 1")
        return v


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class CaseFile(_Strict):
    model: ModelCfg = ModelCfg()
    run_dates: Union[list[RunDate], RunRange, None] = None  # backtests build their own
    calibration: CalibrationCfg | None = None
    data: DataPaths
    sub_basins: dict[str, SubBasin] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, raw):
        """Merge `defaults` under every sub-basin (sub-basin values win)."""
        if isinstance(raw, dict) and "defaults" in raw:
            raw = dict(raw)
            defaults = raw.pop("defaults") or {}
            raw["sub_basins"] = {k: _merge(defaults, v or {}) for k, v in (raw.get("sub_basins") or {}).items()}
        return raw

    @model_validator(mode="after")
    def _consistent(self):
        m = self.model
        if m.assimilate_etp and m.etp != "series":
            raise ValueError("model.assimilate_etp needs model.etp: series")
        if m.etp == "series" and self.data.etp_observed is None:
            raise ValueError("model.etp: series needs data.etp_observed")
        for name, sb in self.sub_basins.items():
            p = sb.params
            if m.runoff == "pmur" and (p.lambda_ is None or p.pmur is None):
                raise ValueError(f"sub_basins.{name}.params: runoff pmur needs lambda and pmur")
            if m.runoff == "ai" and p.ai is None:
                raise ValueError(f"sub_basins.{name}.params: runoff ai needs ai")
            if m.etp == "climatology" and sb.etp_climatology is None:
                raise ValueError(f"sub_basins.{name}: etp climatology needs etp_climatology (12 values)")
            known = set(Params.model_fields) - {"str_", "lambda_"} | {"str", "lambda", "alfa", "ex"} | set(sb.gauges)
            unknown = set(sb.calibration_bounds) - known
            if unknown:
                raise ValueError(f"sub_basins.{name}.calibration_bounds: unknown names {sorted(unknown)}")
            if {"alfa", "ex"} & set(sb.calibration_bounds) and sb.kt_shape is None:
                raise ValueError(f"sub_basins.{name}.calibration_bounds: alfa/ex need kt_shape")
        return self


@dataclass
class Case:
    inputs: Inputs
    config: RunConfig
    spec: CaseFile  # the validated YAML, defaults merged


def _bound_rows(name, var, b):
    if isinstance(b, dict):
        return [(name, f"limite_{side}_{var}", b[m][i], m) for m in range(1, 13)
                for i, side in enumerate(("inferior", "superior"))]
    return [(name, f"limite_{side}_{var}", b[i], 0) for i, side in enumerate(("inferior", "superior"))]


def read_spec(path: str | Path) -> tuple[CaseFile, dict]:
    """Validated case and the raw YAML dict, without reading any series."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CaseFile.model_validate(raw), raw


def param_rows(name: str, sb: SubBasin) -> list[tuple]:
    """Sub-basin parameters as `nome, parametro, valor` rows (what SmapParams.from_table reads)."""
    rows = [(name, "Area", sb.area)]
    rows += [(name, PARAM_NAMES[k], v) for k, v in sb.params.model_dump().items() if v is not None]
    return rows + [(name, "Kt-" + str(-lag) if lag < 0 else f"Kt{lag}", w) for lag, w in sb.kt.items()]


def load_case(path: str | Path, run_dates: pd.DataFrame | None = None) -> Case:
    """Read and validate a YAML case; series paths are relative to the YAML file.

    run_dates (data, numero_dias_previsao) replaces the file's run_dates.
    """
    path = Path(path)
    cf, _ = read_spec(path)
    root = path.parent

    rd = cf.run_dates
    if run_dates is not None:
        dates = run_dates.copy()
    elif rd is None:
        raise ValueError(f"{path}: no run_dates")
    elif isinstance(rd, RunRange):
        days = pd.date_range(str(rd.start), str(rd.end), freq=f"{rd.every}D")
        dates = pd.DataFrame({"data": days, "numero_dias_previsao": rd.horizon})
    else:
        dates = pd.DataFrame({"data": pd.to_datetime([str(r.date) for r in rd]),  # str: same unit as io
                              "numero_dias_previsao": [r.horizon for r in rd]})
    dates = dates.sort_values("data", ignore_index=True)

    params, init, gauges, nc = [], [], [], []
    for name, sb in cf.sub_basins.items():
        params += param_rows(name, sb)
        a = sb.assimilation
        init += [(name, "Ebin", sb.initial.ebin, 0), (name, "Supin", sb.initial.supin, 0),
                 (name, "Tuin", sb.initial.tuin / 100, 0), (name, "numero_dias_assimilacao", a.days, 0),
                 (name, "funcao_objetivo", OBJECTIVE_CODES[a.objective], 0),
                 (name, "ajusta_precipitacao", int(a.adjust_last_day), 0)]
        for var, b in a.bounds.items():
            init += _bound_rows(name, var, b)
        gauges += [(name, g, w) for g, w in sb.gauges.items()]
        if sb.etp_climatology:
            nc += [(m, name, v) for m, v in enumerate(sb.etp_climatology, start=1)]
    gauges = pd.DataFrame(gauges, columns=["nome", "posto", "valor"])

    d = cf.data
    files = {kind: root / p for kind, p in (
        ("VAZAO_OBSERVADA", d.flow_observed), ("PRECIPITACAO_OBSERVADA", d.precip_observed),
        ("PRECIPITACAO_PREVISTA", d.precip_forecast), ("EVAPOTRANSPIRACAO_OBSERVADA", d.etp_observed),
        ("EVAPOTRANSPIRACAO_PREVISTA", d.etp_forecast)) if p is not None}
    if cf.model.etp == "climatology":  # observed ETP series would be unused
        files.pop("EVAPOTRANSPIRACAO_OBSERVADA", None)
        files.pop("EVAPOTRANSPIRACAO_PREVISTA", None)
    basins = list(cf.sub_basins)
    inputs = Inputs(
        parameters=pd.DataFrame(params, columns=["nome", "parametro", "valor"]),
        initialization=pd.DataFrame(init, columns=["nome", "variavel", "valor", "mes"]),
        run_dates=dates, sub_basins=basins,
        etp_climatology=pd.DataFrame(nc, columns=["mes", "nome", "valor"]) if nc else None,
        **_load_series(files, basins, gauges, dates))
    m = cf.model
    return Case(inputs, RunConfig(runoff=m.runoff, etp_source=m.etp, assimilate_etp=m.assimilate_etp,
                                  assimilate=m.assimilate), cf)


def to_yaml(inputs: Inputs, config: RunConfig, path: str | Path) -> Path:
    """Write `inputs` as a YAML case at `path`, with its series as CSVs next to it; returns `path`.

    Rain is written already weighted per sub-basin, so each sub-basin has itself as its only gauge.
    The result is loaded back with `load_case`, so a bad conversion fails here.
    """
    path = Path(path)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)

    def wide(df, fname):
        # shortcut: keeps only dates every sub-basin has; write one case per sub-basin if their spans differ a lot
        df.pivot(index="data", columns="nome", values="valor").dropna().to_csv(
            root / fname, sep=";", date_format="%Y-%m-%d")
        return fname

    def long(df, fname):
        df[["data_rodada", "data_previsao", "cenario", "nome", "valor"]].to_csv(
            root / fname, sep=";", index=False, date_format="%Y-%m-%d")
        return fname

    data = {"precip_observed": wide(inputs.precip_observed, "precipitacao.csv"),
            "flow_observed": wide(inputs.flow_observed, "vazoes.csv"),
            "precip_forecast": long(inputs.precip_forecast, "precipitacao_prevista.csv")}
    if inputs.etp_observed is not None:
        data["etp_observed"] = wide(inputs.etp_observed, "etp.csv")
    if inputs.etp_forecast is not None:
        data["etp_forecast"] = long(inputs.etp_forecast, "evapotranspiracao_prevista.csv")

    yaml_key = {r: Params.model_fields[k].alias or k for k, r in PARAM_NAMES.items()}
    objective = {v: k for k, v in OBJECTIVE_CODES.items()}
    init = inputs.initialization
    sub_basins = {}
    for name in inputs.sub_basins:
        p = inputs.parameters[inputs.parameters["nome"] == name]
        v = dict(zip(p["parametro"], p["valor"].astype(float).tolist()))
        bounds = {}
        for var in ("prec", "ebin"):
            months = {m: [_init_value(init, name, f"limite_{side}_{var}", m) for side in ("inferior", "superior")]
                      for m in range(1, 13)}
            bounds[var] = months[1] if all(b == months[1] for b in months.values()) else months
        sb = {"area": v["Area"],
              "params": {yaml_key[k]: x for k, x in v.items() if k in yaml_key},
              "kt": {int(k[2:]): x for k, x in v.items() if k in KT_NAMES and x > 0},
              "gauges": {name: 1.0},
              "initial": {"ebin": _init_value(init, name, "Ebin"), "supin": _init_value(init, name, "Supin"),
                          "tuin": _init_value(init, name, "Tuin") * 100},
              "assimilation": {
                  "days": int(_init_value(init, name, "numero_dias_assimilacao")),
                  "objective": objective[int(_init_value(init, name, "funcao_objetivo", default=0))],
                  "adjust_last_day": _init_value(init, name, "ajusta_precipitacao", default=0) == 1,
                  "bounds": bounds}}
        if inputs.etp_climatology is not None:
            nc = inputs.etp_climatology[inputs.etp_climatology["nome"] == name].sort_values("mes")
            sb["etp_climatology"] = [float(x) for x in nc["valor"]]
        sub_basins[name] = sb

    spec = {"model": {"runoff": config.runoff, "etp": config.etp_source, "assimilate": config.assimilate,
                      "assimilate_etp": config.assimilate_etp},
            "run_dates": [{"date": pd.Timestamp(d).date(), "horizon": int(h)}
                          for d, h in zip(inputs.run_dates["data"], inputs.run_dates["numero_dias_previsao"])],
            "data": data, "sub_basins": sub_basins}
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_case(path)
    return path
