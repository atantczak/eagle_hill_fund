"""Build (and cache) every FMP alternative-data dataset. Run: python scripts/build_alt_datasets.py"""

import logging
import sys
import time

from ehf.research.data import altdata as A
from ehf.research.data.fmp_provider import FMPProvider
from ehf.research.jobs import load_dotenv_quietly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
load_dotenv_quietly()
end = sys.argv[1] if len(sys.argv) > 1 else "2026-10-30"
names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(A.ALL_DATASETS)
p = FMPProvider()
for n in names:
    t = time.time()
    try:
        b = A.load_dataset(n, p, end)
        print(
            f"OK {n} rows={len(b)} cols={b['symbol'].nunique()} {time.time() - t:.0f}s", flush=True
        )
    except Exception as exc:
        logging.exception("dataset %s failed", n)
        print(f"FAIL {n} {type(exc).__name__}: {exc}", flush=True)
