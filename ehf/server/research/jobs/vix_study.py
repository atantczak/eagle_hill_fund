"""Job: the VIX overlay study (nine rules + vote vs. hold and the 200-day filter).

python -m ehf.server.research.jobs.vix_study                      # full grids, Sharpe objective
python -m ehf.server.research.jobs.vix_study --objective calmar   # drawdown-adjusted objective
python -m ehf.server.research.jobs.vix_study --quick --strategies vix_term,vix_vote
"""

from __future__ import annotations

import argparse
import sys

from ehf.server.research.data.fmp_provider import FMPProvider
from ehf.server.research.data.vix import load_vix_bars
from ehf.server.research.jobs import (
    AlreadyRunning,
    configure_logging,
    load_dotenv_quietly,
    single_instance,
    write_summary,
)
from ehf.server.research.strategies.vix_overlay import GRIDS
from ehf.server.research.studies.vix_overlay import run_study

JOB = "vix_study"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EHF VIX overlay study")
    p.add_argument(
        "--start", default="2006-08-01", help="first evaluation bar (VIX3M exists from 2006-07)"
    )
    p.add_argument("--end", default="2026-10-30")
    p.add_argument("--data-start", default="2004-01-01", help="history start for warm-up")
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument(
        "--objective", default="sharpe", help="metric maximized in-sample (sharpe, calmar, ...)"
    )
    p.add_argument(
        "--strategies", default="all", help=f"comma list or 'all'; known: {', '.join(GRIDS)}"
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="2-point grids (smoke test)")
    p.add_argument(
        "--no-skew", action="store_true", help="skip the Cboe SKEW download (drops vix_skew)"
    )
    p.add_argument("--tag", default=None, help="summary file stamp (default: today)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    load_dotenv_quietly()
    log = configure_logging(JOB, verbose=args.verbose)
    keys = (
        tuple(GRIDS)
        if args.strategies == "all"
        else tuple(s.strip() for s in args.strategies.split(","))
    )
    if args.no_skew:
        keys = tuple(k for k in keys if k != "vix_skew")
    try:
        with single_instance(JOB):
            bars = load_vix_bars(
                FMPProvider(), args.data_start, args.end, include_skew=not args.no_skew
            )
            result = run_study(
                bars,
                start=args.start,
                end=args.end,
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
        "strategies": [o.strategy for o in result.outcomes],
        "verdicts": {o.strategy: o.verdict.status for o in result.outcomes},
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
