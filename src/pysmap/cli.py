"""Command line: `pysmap run <case.yml | folder>` and `pysmap evaluate <previsao.csv> <observed.csv>`."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import forecast, studies
from .case import load_case
from .evaluation import evaluate
from .io import _csv, _wide_history, read_case
from .outputs import write_outputs
from .routing import totalize

MODELS = {"oficial": forecast.OFICIAL, "pmur": forecast.PMUR, "etp": forecast.ETP, "pmur_etp": forecast.PMUR_ETP,
          "pmur_n_assimila_etp": forecast.PMUR_N_ASSIMILA_ETP}


def _run(args):
    src = Path(args.case)
    if src.is_dir():
        inputs = read_case(src)
        # executa_caso_oficial_v2's choice when no model is given
        config = MODELS[args.model] if args.model else (forecast.PMUR_ETP if inputs.etp_forecast is not None
                                                       else forecast.OFICIAL)
    else:
        if args.model:
            raise SystemExit("--model applies to CSV folders; set `model:` in the YAML instead")
        case = load_case(src)
        inputs, config = case.inputs, case.config
    out_dir = Path(args.output) if args.output else (src if src.is_dir() else src.parent) / "saida"
    files, qcalc = set(), []
    for i, (name, out) in enumerate(forecast.iter_sub_basins(inputs, config, args.workers)):
        files.update(write_outputs(out, out_dir, ons_txt=args.ons_txt, append=i > 0))
        if args.totalize:
            qcalc.append(out["previsao"].query("variavel == 'Qcalc'"))
        print(f"{name} done")
    if args.totalize:
        cfg = pd.read_csv(args.totalize, sep=";", encoding="utf-8-sig")
        totalize(pd.concat(qcalc), inputs.flow_observed, cfg).to_csv(out_dir / "totalizada.csv", sep=";",
                                                                     index=False, date_format="%Y-%m-%d")
        files.add(out_dir / "totalizada.csv")
    print(f"wrote {len(files)} files to {out_dir}")


def _evaluate(args):
    prev = _csv(Path(args.forecast))
    for c in ("data_caso", "data_previsao"):
        prev[c] = pd.to_datetime(prev[c])
    prev["valor"] = pd.to_numeric(prev["valor"])
    result = evaluate(prev, _wide_history(Path(args.observed)))
    if args.output:
        result.to_csv(args.output, sep=";", index=False)
        print(f"wrote {args.output}")
    else:
        print(result.to_string(index=False))


def _calibrate(args):
    src = Path(args.case)
    results = studies.calibrate_case(src, args.sub_basin, args.workers)
    out = Path(args.output) if args.output else src.parent / "calibracao"
    files = studies.save_calibration(src, results, out)
    for name, r in results.items():
        print(f"{name}: objective {r.initial_objective:.4f} -> {r.objective:.4f}")
    print(f"wrote {len(files)} files to {out}")


def _validate(args):
    src = Path(args.case)
    out = Path(args.output) if args.output else src.parent / "validacao"
    files = studies.backtest(src, out, args.workers)
    print(f"wrote {len(files)} files to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysmap", description="SMAP/ONS rainfall-runoff model")
    sub = ap.add_subparsers(required=True)
    run = sub.add_parser("run", help="run a case (YAML file or smapOnsR new-format CSV folder)")
    run.add_argument("case")
    run.add_argument("-o", "--output", help="output folder (default: <case dir>/saida)")
    run.add_argument("--ons-txt", action="store_true", help="also write ONS _previsao.txt/_ajuste.txt files")
    run.add_argument("--model", choices=MODELS, help="model variant for CSV folders")
    run.add_argument("-j", "--workers", type=int, help="sub-basins run in parallel processes")
    run.add_argument("--totalize", metavar="CONFIGURACAO_CSV", help="also route flows to stations (totalizada.csv)")
    run.set_defaults(func=_run)
    cal = sub.add_parser("calibrate", help="calibrate sub-basins with calibration_bounds (YAML case)",
                         description="Writes case_calibrated.yml (comments of the original are not kept), "
                                     "calibracao_log.csv, calibracao_serie.csv and calibracao_resumo.csv.")
    cal.add_argument("case")
    cal.add_argument("-o", "--output", help="output folder (default: <case dir>/calibracao)")
    cal.add_argument("--sub-basin", nargs="+", help="only these sub-basins")
    cal.add_argument("-j", "--workers", type=int, help="threads for the finite-difference gradient")
    cal.set_defaults(func=_calibrate)
    val = sub.add_parser("validate", help="backtest: weekly Thursday forecasts over the calibration periods")
    val.add_argument("case")
    val.add_argument("-o", "--output", help="output folder (default: <case dir>/validacao)")
    val.add_argument("-j", "--workers", type=int, help="sub-basins run in parallel processes")
    val.set_defaults(func=_validate)
    ev = sub.add_parser("evaluate", help="forecast skill by lead time")
    ev.add_argument("forecast", help="previsao.csv")
    ev.add_argument("observed", help="observed flow, wide: data;<sub-basin>;...")
    ev.add_argument("-o", "--output", help="write validacao.csv instead of printing")
    ev.set_defaults(func=_evaluate)
    args = ap.parse_args(argv)
    args.func(args)
