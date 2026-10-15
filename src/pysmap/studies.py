"""Case-level studies: calibrate sub-basins from a YAML case, and backtest a case over a period.

Replace smapOnsR's calibration screen (executa_visualizador_calibracao) and executa_validacao_calibracao.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import forecast
from .calibration import CalibrationResult, calibrate
from .case import OBJECTIVE_CODES, Relative, load_case, param_rows, read_spec
from .evaluation import METRICS, evaluate
from .io import _wide_history
from .outputs import write_outputs
from .params import SmapParams

# YAML parameter name -> SmapParams field, where they differ
FIELD = {"str": "str_", "lambda": "lam"}


def _period(p):
    return pd.Timestamp(p.start), pd.Timestamp(p.end)


def calibrate_case(path: str | Path, names: list[str] | None = None,
                   workers: int | None = None) -> dict[str, CalibrationResult]:
    """Calibrate every sub-basin that has `calibration_bounds` (or the ones in `names`)."""
    path = Path(path)
    spec, _ = read_spec(path)
    cal = spec.calibration
    if cal is None:
        raise ValueError(f"{path}: no calibration section")
    names = names or [n for n, sb in spec.sub_basins.items() if sb.calibration_bounds]
    missing = [n for n in names if n not in spec.sub_basins or not spec.sub_basins[n].calibration_bounds]
    if not names or missing:
        raise ValueError(f"no calibration_bounds for {missing or 'any sub-basin'}")
    root = path.parent
    start, end = _period(cal.simulation)
    first, last = _period(cal.objective or cal.simulation)
    days = pd.date_range(start, end)
    scored = days[(days >= first) & (days <= last)]

    weights = np.ones(len(scored))
    for a, b in cal.exclude:
        weights[(scored >= pd.Timestamp(a)) & (scored <= pd.Timestamp(b))] = 0
    if weights.sum() == 0:
        raise ValueError("calibration.exclude removes every scored day")
    weights /= weights.sum()

    flow = _wide_history(root / spec.data.flow_observed, names)
    gauge_ids = sorted({g for n in names for g in spec.sub_basins[n].gauges})
    rain = _wide_history(root / spec.data.precip_observed, gauge_ids).rename(columns={"nome": "posto"})
    etp_obs = _wide_history(root / spec.data.etp_observed, names) if spec.model.etp == "series" else None

    results = {}
    for name in names:
        sb = spec.sub_basins[name]
        p = SmapParams.from_table(pd.DataFrame(param_rows(name, sb), columns=["nome", "parametro", "valor"]))
        shape = (sb.kt_shape.alfa, sb.kt_shape.ex) if sb.kt_shape else None
        current = {**{k: getattr(p, FIELD.get(k, k)) for k in sb.params.model_dump(by_alias=True)},
                   **(dict(zip(("alfa", "ex"), shape)) if shape else {}), **sb.gauges}
        unset = [k for k in sb.calibration_bounds if current.get(k) is None]
        if unset:
            raise ValueError(f"sub_basins.{name}: cannot calibrate {unset}, they have no value")
        bounds = {FIELD.get(k, k): tuple(np.multiply(b.times, current[k])) if isinstance(b, Relative) else b
                  for k, b in sb.calibration_bounds.items()}
        if etp_obs is not None:
            e = etp_obs[etp_obs["nome"] == name].set_index("data")["valor"]
        else:
            e = pd.Series(np.asarray(sb.etp_climatology)[days.month - 1], index=days)
        gauges = pd.DataFrame({"nome": name, "posto": list(sb.gauges), "valor": list(sb.gauges.values())})
        results[name] = calibrate(
            p, rain[rain["posto"].isin(sb.gauges)], gauges, e, flow[flow["nome"] == name].set_index("data")["valor"],
            start, end, sb.initial.ebin, sb.initial.tuin / 100, sb.initial.supin, bounds,
            objective_period=(first, last), kt_shape=shape, pmur=spec.model.runoff == "pmur",
            objective=OBJECTIVE_CODES[cal.metric], weights=weights, eps=cal.step,
            scale_by_upper=cal.scale_by_upper, maxiter=cal.maxiter, workers=workers)
    return results


def calibrated_yaml(path: str | Path, results: dict[str, CalibrationResult]) -> dict:
    """The case YAML with each calibrated sub-basin's params, kt, gauges and kt_shape replaced."""
    spec, raw = read_spec(path)
    raw = copy.deepcopy(raw)
    for name, r in results.items():
        sb = raw["sub_basins"].setdefault(name, {})
        keys = spec.sub_basins[name].params.model_dump(by_alias=True, exclude_none=True)
        sb["params"] = {k: float(getattr(r.params, FIELD.get(k, k))) for k in keys}
        sb["kt"] = {2 - i: float(w) for i, w in enumerate(r.params.kt) if w > 0}  # KT_NAMES order: lag 2 first
        sb["gauges"] = {g: float(w) for g, w in zip(r.gauges["posto"], r.gauges["valor"])}
        if r.kt_shape:
            sb["kt_shape"] = {"alfa": float(r.kt_shape[0]), "ex": float(r.kt_shape[1])}
    return raw


def save_calibration(path: str | Path, results: dict[str, CalibrationResult], out_dir: str | Path) -> list[Path]:
    """Write case_calibrated.yml, calibracao_log.csv, calibracao_serie.csv and calibracao_resumo.csv."""
    path, out = Path(path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec, _ = read_spec(path)
    calibrated = calibrated_yaml(path, results)
    # data paths are relative to the case file; keep them valid from the output folder
    calibrated["data"] = {k: str((path.parent / v).resolve()) for k, v in calibrated["data"].items()}
    files = [out / f for f in ("case_calibrated.yml", "calibracao_log.csv", "calibracao_serie.csv",
                               "calibracao_resumo.csv")]
    files[0].write_text(yaml.safe_dump(calibrated, sort_keys=False, allow_unicode=True), encoding="utf-8")

    flow = _wide_history(path.parent / spec.data.flow_observed, list(results))
    logs, series, summary = [], [], []
    for name, r in results.items():
        logs.append(r.log.assign(nome=name))
        obs = flow[flow["nome"] == name].set_index("data")["valor"].reindex(r.simulation.index)
        s = pd.DataFrame({"data": r.simulation.index, "nome": name, "observado": obs.to_numpy(),
                          "simulado": r.simulation.to_numpy(),
                          "peso": r.weights.reindex(r.simulation.index).fillna(0).to_numpy()})
        series.append(s)
        sc = s[s["data"].isin(r.weights.index)]
        w = sc["peso"].to_numpy()
        summary.append({"nome": name, "objetivo_inicial": r.initial_objective, "objetivo_final": r.objective,
                        **{m: f(sc["simulado"].to_numpy(), sc["observado"].to_numpy(), w) for m, f in METRICS.items()}})
    pd.concat(logs, ignore_index=True).to_csv(files[1], sep=";", index=False)
    pd.concat(series, ignore_index=True).to_csv(files[2], sep=";", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(summary).to_csv(files[3], sep=";", index=False)
    return files


def thursdays(start, end, horizon: int) -> pd.DatetimeIndex:
    """Backtest run dates, as R's cria_datas: from the first Thursday on or after `start`, weekly,
    or monthly when the horizon is longer than 42 days."""
    start = pd.Timestamp(start)
    first = start + pd.Timedelta(days=(3 - start.dayofweek) % 7)
    return pd.date_range(first, end, freq=pd.DateOffset(months=1) if horizon > 42 else "7D")


def backtest(path: str | Path, out_dir: str | Path, workers: int | None = None) -> list[Path]:
    """Chained forecasts every Thursday of the calibration simulation period, scored over its objective
    period. Writes the run tables plus validacao.csv to `out_dir`."""
    path, out = Path(path), Path(out_dir)
    spec, _ = read_spec(path)
    cal = spec.calibration
    if cal is None:
        raise ValueError(f"{path}: no calibration section")
    first, last = _period(cal.objective or cal.simulation)
    dates = pd.DataFrame({"data": thursdays(cal.simulation.start, cal.simulation.end, cal.horizon),
                          "numero_dias_previsao": cal.horizon})
    case = load_case(path, run_dates=dates)
    # every run needs its assimilation window observed, and every forecast day observed to be scored.
    # R used min(flow) - n here, which lets runs start before the data does.
    flow = case.inputs.flow_observed
    n = max(sb.assimilation.days for sb in spec.sub_basins.values())
    dates = dates[(dates["data"] >= flow["data"].min() + pd.Timedelta(days=n))
                  & (dates["data"] <= flow["data"].max() - pd.Timedelta(days=cal.horizon - 1))]
    if dates.empty:
        raise ValueError("no backtest date has enough observed flow around it")
    case.inputs.run_dates = dates.reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    files = set()
    for i, (_, res) in enumerate(forecast.iter_sub_basins(case.inputs, case.config, workers)):
        res = {k: v[(v["data_caso"] >= first) & (v["data_caso"] <= last)] for k, v in res.items()}
        files.update(write_outputs(res, out, append=i > 0))
        evaluate(res["previsao"], flow).to_csv(out / "validacao.csv", sep=";", index=False,
                                                mode="a" if i else "w", header=not i)
    files.add(out / "validacao.csv")
    return sorted(files)
