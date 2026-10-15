# pysmap

Python port of the ONS SMAP/ONS rainfall-runoff model ([ONSBR/smapOnsR](https://github.com/ONSBR/smapOnsR) and its
C++ kernels in [ONSBR/funcaoSmapCpp](https://github.com/ONSBR/funcaoSmapCpp)).

```bash
uv sync
uv run pytest            # add -m "not slow" to skip the 21-sub-basin golden case
```

## Running a case

A case is one YAML file: model variant, forecast dates, per-sub-basin parameters, Kt weights, gauges, initial
state and assimilation bounds, with shared values under `defaults`. Time series stay in CSV/Parquet files it
points to. See [`examples/case.yml`](examples/case.yml); errors name the offending field.

```bash
uv run pysmap run examples/case.yml -o out --ons-txt     # previsao/otimizacao/funcao_objetivo/ajuste.csv + ONS .txt
uv run pysmap run tests/data/arq_entrada_pmur            # smapOnsR new-format CSV folder also works (--model ...)
uv run pysmap run case.yml --totalize configuracao.csv   # also route to stations: totalizada.csv
uv run pysmap run case.yml -j 8                         # 8 sub-basins at once; results stream to disk
uv run pysmap evaluate saida/previsao.csv vazoes.csv     # PBIAS/NSE/MAPE/DM/KGE/RMSE by daily, weekly, 30-day lead
uv run pysmap calibrate case.yml -j 4                    # fit sub-basins with calibration_bounds -> case_calibrated.yml
uv run pysmap validate case.yml                          # backtest: Thursday forecasts over the calibration period
```

```python
import pysmap

case = pysmap.load_case("examples/case.yml")
out = pysmap.run_chained(case.inputs, case.config)

pysmap.write_outputs(out, "out", ons_txt=True)
pysmap.evaluate(out["previsao"], case.inputs.flow_observed)
```

## Layout

| Module | Replaces in smapOnsR |
|---|---|
| `params` | `new_modelo_smap_ons[_pmur]`, `inicializacao_smap`: named `SmapParams` instead of `unlist(modelo)[75]` indices |
| `kernel` | the 4 C++ kernels (single/multi-scenario × Ai/PMUR) as one Numba function |
| `weighting` | `ponderacao_temporal[_cpp/_cenario_cpp]`, `ponderacao_espacial[_previsao]` |
| `metrics` | `metricas.R` (`OBJECTIVES` maps the `funcao_objetivo` code) |
| `assimilation` | the 4 `assimilacao_*` functions and their objective functions |
| `forecast` | the 6 `rodada_encadeada_*` / `rodada_sem_assimilacao` functions, chosen by `RunConfig` |
| `routing` | `totaliza_previsao`, `propaga_tv`, `propaga_muskingum`, `ordem_afluencia` |
| `calibration` | `calibracao`, `calibracao[_pmur]_paralela`, `cria_kt` |
| `io` | `le_arq_entrada_novo` and the readers it uses |
| `case` | YAML case files (new; replaces arquivos/parametros/inicializacao/postos_plu/datasRodadas CSVs) |
| `outputs` | `escreve_previsao`, `escreve_ajuste` and the CSV writes of `executa_caso_oficial_v2` |
| `evaluation` | `analisa_previsoes`, `agrega_semanal`, `agrega_mensal` |
| `cli` | `executa_caso_*` as `pysmap run` / `pysmap evaluate` |

`RunConfig` presets: `OFICIAL` (Ai, monthly ETP climatology), `PMUR`, `ETP` (Ai, observed+forecast ETP with ETP
multipliers assimilated), `PMUR_ETP` (what `executa_caso_oficial_v2` runs), `PMUR_N_ASSIMILA_ETP`.
`RunConfig(assimilate=False)` runs open-loop from the given initial state.

Calibration takes bounds by name; parameters without bounds stay fixed, so any subset can be calibrated.
In a case file, `calibration:` sets the simulation/objective periods, excluded periods, metric (dm, nse, mape,
kge) and optimizer options; each sub-basin's `calibration_bounds` (absolute `[lo, hi]` or `{times: [lo, hi]}`)
and optional `kt_shape` say what is fitted. `pysmap calibrate` writes the calibrated case plus an iteration
log, the observed-vs-simulated series and a metric summary; `pysmap validate` runs the backtest that
smapOnsR's `executa_validacao_calibracao` did. The backtest drops run dates whose assimilation window or
forecast lacks observed flow (R allowed runs starting before the data).
`workers=N` computes the gradient on N threads.

## Parity with R

- Kernel and weighting reproduce R exactly (sobradinho 2-day test: 1100, 1065.9265).
- Optimization uses scipy's L-BFGS-B with R `optim`'s settings (lmm=5, factr=1e7, pgtol=0, `ndeps` central
  differences clipped to the bounds, `parscale`). It is a different implementation, so the optimum differs
  slightly. On the PMUR+ETP golden case, stoantjari's day-35 forecast is 1355.46 against R's 1352.18 (0.24%).
- Kept on purpose: the per-variant `ndeps` and `parscale` values, the zero values in the first days of routing,
  and applying the assimilated rain weight only up to day d-2.
- Changed:
  - `Ed2`/`Ed3` follow the C++ naming everywhere.
  - Per-sub-basin `funcao_objetivo` is read correctly. R's `pmur_etp` did not filter it by sub-basin.
  - `fill_forecast` fills each run date with its own mean.
  - Observed flow from day d onward is ignored when totalizing.
  - Regressed stations wait for their base station in the routing order.
  - The forecast loop raises a clear error on gaps or overlaps instead of misaligning arrays.
  - Evaluation joins forecast and observation by date and scores each scenario separately. R matched by row
    position and averaged scenarios together in its weekly and monthly blocks; results agree for one scenario
    with no missing days, and R's golden values reproduce.

Not ported: the Shiny visualizers, the old fixed-width input reader (`leitura_antiga.R`), the calibration
screen's plots, a CSV→YAML converter, and reading back previous outputs (`leitura_saida.R`).
