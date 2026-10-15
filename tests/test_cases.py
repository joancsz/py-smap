"""Full chained runs on smapOnsR's new-format fixture cases, against the values its tests hard-code."""
import time

import numpy as np
import pytest

from conftest import DATA
from pysmap import forecast
from pysmap.io import read_case

# test-rodadas_encadeadas.R "testa rodada com aprimoramentos" (Windows R build), one value per sub-basin
PMUR_FO = [0.1985781, 0.0874264, 0.0006976, 0.4497266, 0.2600375, 0.7954830, 0.0425972, 0.2157436, 0.0807382,
           0.0774346, 0.1810360, 0.1462257, 0.3211804, 0.2018538, 0.2933669, 0.1392731, 0.1794551, 0.1521796,
           0.1845175, 0.0144009, 0.0623383]


@pytest.mark.slow
def test_pmur_etp_case():
    inputs = read_case(DATA / "arq_entrada_pmur")
    t = time.perf_counter()
    out = forecast.run_chained(inputs, forecast.PMUR_ETP)
    print(f"\n{len(inputs.sub_basins)} sub-basins in {time.perf_counter() - t:.1f}s")
    prev = out["previsao"]
    q = prev[(prev.nome == "stoantjari") & (prev.variavel == "Qcalc") & (prev.cenario == "ecmwf_7")].valor
    print("stoantjari ecmwf_7 day 35:", q.iloc[34])
    assert q.iloc[34] == pytest.approx(1352.18, rel=0.01)
    # scipy's L-BFGS-B is not R's: allow a different optimum, but never a clearly worse fit
    fo, r = out["funcao_objetivo"].funcao_objetivo.to_numpy(), np.array(PMUR_FO)
    assert (fo <= np.maximum(r * 1.05, r + 0.005)).all()
