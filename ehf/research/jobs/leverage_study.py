"""Job: the leverage study (3x fund when the market turns bullish, plain fund otherwise).

python -m ehf.research.jobs.leverage_study                         # every pair, both variants
python -m ehf.research.jobs.leverage_study --pairs QQQ/TQQQ --quick
python -m ehf.research.jobs.leverage_study --variants extended --strategies lev_sma,lev_swing
"""

from __future__ import annotations

import argparse
import sys

from ehf.research.data.fmp_provider import FMPProvider
from ehf.research.data.leverage import PAIRS, Pair, load_pair_bars
from ehf.research.jobs import (
    AlreadyRunning,
    configure_logging,
    load_dotenv_quietly,
    single_instance,
    write_summary,
)
from ehf.research.strategies.leverage import GRIDS
from ehf.research.studies.leverage import run_study

JOB = "leverage_study"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EHF leverage study")
    p.add_argument("--pairs", default="all", help=f"comma list or 'all'; known: {', '.join(PAIRS)}")
    p.add_argument("--start", default="2000-03-01", help="earliest evaluation bar (extended runs)")
    p.add_argument("--end", default="2026-10-31")
    p.add_argument("--data-start", default="1998-01-01")
    p.add_argument("--variants", default="extended,real")
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--objective", default="sharpe")
    p.add_argument(
        "--strategies", default="all", help=f"comma list or 'all'; known: {', '.join(GRIDS)}"
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="2-point grids (smoke test)")
    p.add_argument("--tag", default=None, help="summary file stamp (default: today)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    load_dotenv_quietly()
    log = configure_logging(JOB, verbose=args.verbose)
    pairs = (
        tuple(PAIRS)
        if args.pairs == "all"
        else tuple(s.strip().upper() for s in args.pairs.split(","))
    )
    keys = (
        tuple(GRIDS)
        if args.strategies == "all"
        else tuple(s.strip() for s in args.strategies.split(","))
    )
    variants = tuple(v.strip() == "extended" for v in args.variants.split(",") if v.strip())
    provider = FMPProvider()

    def load(pair: Pair):  # type: ignore[no-untyped-def]
        return load_pair_bars(provider, pair, args.data_start, args.end)

    try:
        with single_instance(JOB):
            result = run_study(
                load,
                pairs=pairs,
                start=args.start,
                end=args.end,
                variants=variants,
                strategies=keys,
                train_months=args.train_months,
                test_months=args.test_months,
                objective=args.objective,
                quick=args.quick,
                n_workers=args.workers,
                tag=args.tag,
            )
    except AlreadyRunning as exc:
        log.warning("%s", exc)
        return 2
    summary = {
        "objective": args.objective,
        "pairs": list(result.pairs),
        "outcomes": len(result.outcomes),
        "verdicts": {f"{o.pair}:{o.strategy}": o.verdict.status for o in result.outcomes},
        "failures": result.failures,
        "summary_md": str(result.summary_md),
        "summary_json": str(result.summary_json),
    }
    write_summary(JOB, summary)
    print(result.table().round(3).to_string())
    print(f"\nsummary -> {result.summary_md}")
    return 1 if result.failures and not result.outcomes else 0


if __name__ == "__main__":
    sys.exit(main())
