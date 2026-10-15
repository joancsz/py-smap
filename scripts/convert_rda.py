"""One-off: convert smapOnsR's data/*.rda into gzipped CSV test fixtures (needs the dev group)."""
import sys
from pathlib import Path

import pyreadr

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted(src.glob("*.rda")):
    for name, df in pyreadr.read_r(f).items():
        df.to_csv(dst / f"{name}.csv.gz", index=False)
        print(name, df.shape, list(df.columns))
