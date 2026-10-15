"""SMAP/ONS sub-basin parameters and initial state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Kt2, Kt1 (future days), Kt0 (today), Kt-1 .. Kt-60 (past days): the 63 temporal weights.
KT_NAMES = ["Kt2", "Kt1", "Kt0"] + [f"Kt-{i}" for i in range(1, 61)]


@dataclass(frozen=True)
class SmapParams:
    name: str
    area: float
    str_: float  # soil reservoir capacity (Str); `str` would shadow the builtin
    k2t: float
    crec: float
    capc: float
    k_kt: float
    h1: float
    k2t2: float
    h: float
    k1t: float
    k3t: float
    pcof: float
    ecof: float
    ecof2: float
    kt: np.ndarray
    ai: float | None = None  # fixed initial abstraction (classic SMAP/ONS)
    lam: float | None = None  # Lambda: Ai as a fraction of free soil storage (PMUR variant)
    pmur: float | None = None  # % of field capacity under which real ET stops (PMUR variant)

    @property
    def kt_max(self) -> int:
        """Number of future days weighted (positive Kt2/Kt1)."""
        return int((self.kt[:2] > 0).sum())

    @property
    def kt_min(self) -> int:
        """Furthest past lag with a positive weight."""
        pos = np.flatnonzero(self.kt[3:] > 0)
        return int(pos[-1] + 1) if pos.size else 0

    @classmethod
    def from_table(cls, df: pd.DataFrame, name: str | None = None) -> SmapParams:
        """Build from the long table `nome;parametro;valor` (one sub-basin, or pick `name`)."""
        if name is not None:
            df = df[df["nome"] == name]
        names = df["nome"].unique()
        if len(names) != 1:
            raise ValueError(f"expected one sub-basin, got {list(names)}")
        v = dict(zip(df["parametro"], df["valor"].astype(float)))
        missing = [k for k in ("Area", "Str", "K2t", "Crec", "Capc", "K_kt", "H1", "K2t2", "H", "K1t",
                               "K3t", "Pcof", "Ecof", "Ecof2") if k not in v]
        if missing:
            raise ValueError(f"sub-basin {names[0]} lacks parameters {missing}")
        return cls(
            name=names[0], area=v["Area"], str_=v["Str"], k2t=v["K2t"], crec=v["Crec"], capc=v["Capc"],
            k_kt=v["K_kt"], h1=v["H1"], k2t2=v["K2t2"], h=v["H"], k1t=v["K1t"], k3t=v["K3t"],
            pcof=v["Pcof"], ecof=v["Ecof"], ecof2=v["Ecof2"],
            kt=np.array([v.get(k, 0.0) for k in KT_NAMES]),
            ai=v.get("Ai"), lam=v.get("Lambda"), pmur=v.get("Pmur"),
        )


def initial_state(p: SmapParams, ebin: float, tuin: float, supin: float) -> np.ndarray:
    """Reservoir levels [Rsolo, Rsup, Rsup2, Rsub] from base flow, soil moisture and surface flow."""
    if supin < 0:
        raise ValueError("Supin must be >= 0")
    if not 0 <= tuin <= 1:
        raise ValueError("Tuin must be between 0 and 1")
    if ebin < 0:
        raise ValueError("Ebin must be >= 0")
    rsup = supin / (1 - 0.5 ** (1 / p.k2t)) * 86.4 / p.area
    rsub = ebin / (1 - 0.5 ** (1 / p.k_kt)) * 86.4 / p.area
    return np.array([p.str_ * tuin, rsup, 0.0, rsub])
