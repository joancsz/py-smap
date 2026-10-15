from functools import cache
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent / "data"


@cache
def _load(name):
    df = pd.read_csv(DATA / f"{name}.csv.gz")
    if "data" in df:
        df["data"] = pd.to_datetime(df["data"])
    return df


@pytest.fixture
def rdata():
    """smapOnsR's bundled datasets (data/*.rda), by their R names.

    Made once with pyreadr: `for name, df in pyreadr.read_r(f).items(): df.to_csv(f"{name}.csv.gz", index=False)`.
    """
    return _load
