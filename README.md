# pysmap

Python port of the ONS **SMAP/ONS** rainfall-runoff model, used by Brazil's grid operator (ONS) for daily
inflow forecasts. It reproduces [ONSBR/smapOnsR](https://github.com/ONSBR/smapOnsR) and its C++ kernels in
[ONSBR/funcaoSmapCpp](https://github.com/ONSBR/funcaoSmapCpp) as one pure-Python package. The model kernel is
compiled with Numba, and no R toolchain is needed.

It reads the same inputs as the R package: ONS's legacy fixed-width deck, smapOnsR's "new format" CSV folder,
and a new YAML case file. It writes the same outputs: `;` CSVs and the ONS `_previsao.txt` / `_ajuste.txt`
files.

## Quickstart

### With uv

```bash
git clone <this repo> && cd py-smap
uv sync                                          # Python >= 3.11; installs numpy, pandas, scipy, numba, pydantic
uv run pysmap run examples/case.yml -o out --ons-txt
```

`out/` now holds `previsao.csv` (forecast flows and states), `ajuste.csv` (the assimilation fit),
`otimizacao.csv` (fitted multipliers), `funcao_objetivo.csv`, plus `<sub-basin>_<scenario>_previsao.txt` and
`<sub-basin>_ajuste.txt` in the ONS layout. The first run takes a few seconds longer while Numba compiles the
kernel; later runs use the cache.

Run the tests with `uv run pytest`. Add `-m "not slow"` to skip the full golden cases (about 30 s without them).

### With Docker

```bash
docker build -t pysmap .
docker run --rm pysmap run examples/case.yml -o /tmp/out          # bundled example
docker run --rm -v "$PWD/caso:/data" pysmap run /data             # your ONS case: /data/Arq_Entrada -> /data/Arq_Saida
docker run --rm -v "$PWD:/work" pysmap run /work/case.yml -o /work/saida
```

The image's entrypoint is `pysmap`, so anything after the image name is a CLI command.

### From Python

```python
import pysmap

case = pysmap.load_case("examples/case.yml")        # or pysmap.read_case(folder) / pysmap.read_legacy_case(folder)
out = pysmap.run_chained(case.inputs, case.config)  # dict of DataFrames: previsao, otimizacao, funcao_objetivo, assimilacao

pysmap.write_outputs(out, "out", ons_txt=True)
pysmap.evaluate(out["previsao"], case.inputs.flow_observed)
```

## How the model works

For each **sub-basin** and **forecast date d**, pysmap does three things:

1. **Rain weighting.** Gauge (or grid) rain is averaged into one series per sub-basin with the gauge weights.
   It is then smoothed in time with the **Kt** weights: lags 2 and 1 are future days, 0 is today, and
   −1…−60 are past days.
2. **Assimilation.** Over the `n` days before d (31 in the official deck), it fits one rain multiplier per
   day, the initial base flow (Ebin) and surface flow (Supin), and optionally daily ETP multipliers. The goal
   is to make simulated flow match observed flow. The objective is DM (distance to MAPE = 0, NSE = 1), NSE,
   MAPE or KGE, weighted towards recent days, and solved with bounded L-BFGS-B.
3. **Forecast.** From the assimilated state it simulates every rain **scenario** over the horizon (for
   example 19 or 42 days). It then hot-starts the next forecast date from that state ("chained" runs).

The water balance is SMAP's reservoir model: soil (Rsolo), surface (Rsup), second surface (Rsup2) and
groundwater (Rsub). Its outputs are surface, plan-overflow and base flows (Qsup1, Qsup2, Qplan, Qbase), which
add up to Qcalc. Two runoff variants are supported:

- **Ai** (classic SMAP/ONS): a fixed initial abstraction.
- **PMUR**: the abstraction follows free soil storage (Lambda), and real ET stops below a soil-moisture
  threshold (Pmur).

ETP comes either from a monthly **climatology** or from **observed + forecast series**.

`RunConfig` picks the variant. These presets match the R functions they replace:

| Preset | Runoff | ETP | Replaces |
|---|---|---|---|
| `OFICIAL` | Ai | monthly climatology | `rodada_encadeada_oficial` (`executa_caso_oficial`) |
| `ETP` | Ai | series, ETP multipliers assimilated | `rodada_encadeada_etp` |
| `PMUR` | PMUR | monthly climatology | `rodada_encadeada_pmur` |
| `PMUR_ETP` | PMUR | series, ETP multipliers assimilated | `rodada_encadeada_pmur_etp` (`executa_caso_oficial_v2`) |
| `PMUR_N_ASSIMILA_ETP` | PMUR | series, not assimilated | `rodada_encadeada_pmur_nAssimilaEtp` |

`RunConfig(assimilate=False)` runs open-loop from the given initial state (`rodada_sem_assimilacao`).

## Inputs

pysmap accepts three input formats. All three load into the same `Inputs` object of long DataFrames, so
everything downstream is shared. Every reader validates its input and names the offending file or field.
It checks for gaps, duplicates, negative values, Kt and gauge weights that don't sum to 1, missing
sub-basins, and missing scenarios.

| Format | What it is | Load with |
|---|---|---|
| **YAML case** | One readable file for the model variant, run dates, per-sub-basin parameters, Kt, gauges, initial state and assimilation bounds, with shared values under `defaults`. Time series stay in CSV/Parquet files it points to. See [`examples/case.yml`](examples/case.yml). | `pysmap run case.yml`, `load_case` |
| **smapOnsR new format** | A folder with `arquivos.csv` that indexes `;` CSVs: parameters, initialization, `datasRodadas`, flow, rain, ETP and forecasts. | `pysmap run folder/ [--model ...]`, `read_case` |
| **ONS legacy deck** | `Arq_Entrada/` with `CASO.txt`, `bat.conf`, `MODELOS_PRECIPITACAO.txt` and per-sub-basin fixed-width `_PARAMETROS`, `_INICIALIZACAO`, `_EVAPOTRANSPIRACAO`, `_POSTOS_PLU`, flow and gauge `_c.txt` files. Grid-point rain forecasts can use any of the 3 `arqPrev` layouts. | `pysmap run caso/`, `read_legacy_case` |

`pysmap convert` turns a CSV folder or a legacy deck into a YAML case plus its CSVs, so you can edit it from
then on.

## Command line

```bash
pysmap run examples/case.yml -o out --ons-txt      # YAML case; --ons-txt adds the ONS .txt files
pysmap run tests/data/arq_entrada_pmur             # new-format folder; model guessed as R does, or --model pmur_etp
pysmap run caso/                                   # legacy: caso/Arq_Entrada -> caso/Arq_Saida (overwrites, like R)
pysmap run case.yml -j 8                           # 8 sub-basins in parallel processes; results stream to disk
pysmap run case.yml --totalize configuracao.csv    # also route to stations -> totalizada.csv
pysmap convert caso/Arq_Entrada -o c/case.yml      # any folder -> YAML + CSVs (validated by loading it back)
pysmap evaluate saida/previsao.csv vazoes.csv      # PBIAS/NSE/MAPE/DM/KGE/RMSE by daily, weekly, 30-day lead
pysmap calibrate case.yml -j 4                     # fit parameters -> calibracao/case_calibrated.yml + logs
pysmap validate case.yml                           # backtest: weekly Thursday forecasts over the calibration period
```

### Calibration and validation

Calibration takes bounds by name. Parameters without bounds stay fixed, so you can calibrate any subset. In a
case file:

- `calibration:` sets the simulation and objective periods, excluded periods, the metric (dm, nse, mape or
  kge), and the optimizer step, scaling and iteration limit.
- Each sub-basin's `calibration_bounds` says what is fitted, as absolute `[lo, hi]` or relative
  `{times: [lo, hi]}`. It can include gauge weights.
- Each sub-basin's optional `kt_shape` makes the Kt weights a fitted Beta shape (`alfa`, `ex`), as R's
  `cria_kt` does.

`pysmap calibrate` writes the calibrated case (comments are not kept), an iteration log, the
observed-vs-simulated series and a metric summary. `-j N` computes the gradient on N threads.

`pysmap validate` runs the backtest from smapOnsR's `executa_validacao_calibracao`. It forecasts every
Thursday (monthly when the horizon is over 42 days) and scores the forecasts by lead time. It skips run dates
whose assimilation window or forecast lacks observed flow.

### Totalization

`--totalize configuracao.csv` turns the sub-basins' incremental forecasts into total flow at each station of
ONS's `configuracao.csv`, working upstream first. Flow is routed downstream by travel time or Muskingum.
Regressed stations take another station's flow times a regression factor. The 90 days of observed flow
before each run date warm up the routing. The output has incremental, total, total-without-delay and
delayed-incremental flows per station.

## Code structure

```
src/pysmap/
  params.py        SmapParams: named parameters + Kt; initial reservoir state from Ebin/Supin/Tuin
  kernel.py        the daily water balance (Numba); one function for single/multi-scenario x Ai/PMUR
  weighting.py     temporal (Kt) and spatial (gauge) rain weighting
  metrics.py       weighted NSE, MAPE, DM, PBIAS, KGE, RMSE, correlation; OBJECTIVES maps funcao_objetivo codes
  assimilation.py  fits rain/ETP multipliers + Ebin/Supin before each forecast (R optim's L-BFGS-B settings)
  forecast.py      RunConfig presets, Inputs, chained runs per sub-basin (optionally in parallel)
  routing.py       travel-time and Muskingum propagation, station totalization
  calibration.py   parameter calibration against observed flow; Kt from a Beta shape
  studies.py       case-level calibration (writes the calibrated YAML) and Thursday backtests
  evaluation.py    forecast skill by lead time (daily, weekly, 30-day blocks)
  io.py            smapOnsR new-format CSV folder reader; forecast gap filling
  legacy.py        ONS legacy Arq_Entrada deck reader
  case.py          YAML case schema (pydantic), load_case, to_yaml converter
  outputs.py       CSV and ONS .txt writers, and readers for the .txt files
  cli.py           the pysmap command
tests/             pytest suite; tests/data holds smapOnsR's golden inputs and outputs
examples/case.yml  annotated YAML case
Dockerfile         CLI image
```

How the modules depend on each other: `cli` → readers (`case` / `io` / `legacy`) → `Inputs` → `forecast`
(`weighting` → `assimilation` → `kernel`) → `outputs`, with `routing` and `evaluation` used after a run.

### Mapping to smapOnsR

| Module | Replaces in smapOnsR |
|---|---|
| `params` | `new_modelo_smap_ons[_pmur]`, `inicializacao_smap`: named `SmapParams` instead of `unlist(modelo)[75]` indices |
| `kernel` | the 4 C++ kernels (single/multi-scenario × Ai/PMUR) as one Numba function |
| `weighting` | `ponderacao_temporal[_cpp/_cenario_cpp]`, `ponderacao_espacial[_previsao]` |
| `metrics` | `metricas.R` |
| `assimilation` | the 4 `assimilacao_*` functions and their objective functions |
| `forecast` | the 6 `rodada_encadeada_*` / `rodada_sem_assimilacao` functions, chosen by `RunConfig` |
| `routing` | `totaliza_previsao`, `propaga_tv`, `propaga_muskingum`, `ordem_afluencia` |
| `calibration` | `calibracao`, `calibracao[_pmur]_paralela`, `cria_kt` |
| `studies` | the calibration screen (`executa_visualizador_calibracao`), `executa_validacao_calibracao` |
| `evaluation` | `analisa_previsoes`, `agrega_semanal`, `agrega_mensal` |
| `io` | `le_arq_entrada_novo` and its readers, `valida_*`, `completa_previsao`, `transforma_historico_previsao` |
| `legacy` | `le_arq_entrada` and the `le_entrada_*` readers (`leitura_antiga.R`) |
| `case` | new: YAML cases replacing the arquivos/parametros/inicializacao/postos_plu/datasRodadas CSVs; `to_yaml` converter |
| `outputs` | `escreve_previsao`, `escreve_ajuste`, the CSV writes of `executa_caso_oficial_v2`; `le_previsao`, `le_ajuste` |
| `cli` | `executa_caso_oficial`, `executa_caso_novo`, `executa_caso_oficial_v2` as `pysmap run`, plus the other commands |

**Not ported:** the Shiny visualizers and the calibration screen's plots, since `calibrate` writes the series
for you to plot. Also not ported are `le_execucao` and `le_previsao_2`, which read the old official program's
`execucao.txt` and single-file `previsao.txt`; pysmap never writes either file.

## Parity with R

- Kernel and weighting reproduce R exactly (sobradinho 2-day test: 1100, 1065.9265).
- Optimization uses scipy's L-BFGS-B with R `optim`'s settings (lmm=5, factr=1e7, pgtol=0, `ndeps` central
  differences clipped to the bounds, `parscale`). It is a different implementation, so the optimum differs
  slightly. On the PMUR+ETP golden case, stoantjari's day-35 forecast is 1355.46 against R's 1352.18 (0.24%).
- Legacy deck (`tests/data/legacy`, 22 sub-basins): P_da_Beira matches the official output within 0.04%
  (3492.81 against 3491.56; R gives 3493). The others are within 6%. The official program fits with a
  particle swarm, which reaches the same assimilation fit with a different base/surface flow split.
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
    position and averaged scenarios together in its weekly and monthly blocks. Results agree for one scenario
    with no missing days, and R's golden values reproduce.
  - Rain weighting raises an error when a gauge has no value on a day. R summed the gauges that were there
    and silently undercounted the rain.
  - Calibration replaces zero observed flow with 0.1, as assimilation does, so MAPE/DM stay finite on dry days.
  - Monthly evaluation blocks average forecast and observation over the same days. R kept forecast days that
    had no observation.
  - With several run dates, `_ajuste.txt` is written once per run (`<nome>_<YYYYMMDD>_ajuste.txt`), as the
    forecast files already were. R appended all runs to one file.
  - The legacy reader treats `-` in flow files as a missing value and rejects only real gaps. R stopped when
    more than one such row appeared, even at the end of the series.
- `convert` writes rain already weighted per sub-basin, so each sub-basin is its own gauge. It keeps only the
  dates every sub-basin has.
