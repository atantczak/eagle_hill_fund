"""Job: the grand sweep — a day-long, resumable queue of walk-forward studies.

python -m ehf.research.jobs.grand_sweep --tag 2026-11-04 --hours 10
scripts/run_sweep_shards.sh 2026-11-04 10 3 --hours 9   # ten queues side by side
python -m ehf.research.jobs.grand_sweep --tag smoke --max-tier 0 --strategies tsmom
python -m ehf.research.jobs.grand_sweep --tag 2026-11-04 --plan-only   # count items, run nothing

Outputs under ``var/sweeps/<tag>/``: ``leaderboard.md`` (rewritten after every
item), ``results.jsonl``/``results.csv`` (one row per study), ``robustness.csv``,
``scorecards/`` (one card per study) and a private ``results/`` registry.
Re-running with the same tag resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from ehf.research.data.fmp_provider import FMPProvider
from ehf.research.jobs import (
    AlreadyRunning,
    configure_logging,
    load_dotenv_quietly,
    single_instance,
    write_summary,
)
from ehf.research.studies.grand_sweep import build_plan, run_sweep_queue

JOB = "grand_sweep"


def _csv(v: str | None) -> list[str] | None:
    return [s.strip() for s in v.split(",") if s.strip()] if v else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EHF grand sweep")
    p.add_argument("--tag", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    p.add_argument("--end", default="2026-10-30")
    p.add_argument("--hours", type=float, default=None, help="stop starting new items after this")
    p.add_argument("--max-tier", type=int, default=99, help="0 = canonical settings only")
    p.add_argument("--strategies", default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--families", default="basket,dial")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1, help="run this many queues side by side")
    p.add_argument("--shard-base", type=int, default=0, help="offset for this batch's queue ids")
    p.add_argument("--export", action="store_true", help="copy finished studies to the site")
    p.add_argument("--leaderboard", action="store_true", help="rewrite the leaderboard only")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    load_dotenv_quietly()
    log = configure_logging(JOB, verbose=args.verbose)
    plan = build_plan(
        strategies=_csv(args.strategies),
        targets=_csv(args.targets),
        families=_csv(args.families) or ["basket", "dial", "alt"],
        max_tier=args.max_tier,
    )
    if args.export:
        from ehf.research.studies.grand_sweep import export_to_main

        print(export_to_main(args.tag))
        return 0
    if args.leaderboard:
        from ehf.research.studies.grand_sweep import SweepState, write_leaderboard

        print(write_leaderboard(SweepState.load(args.tag), plan))
        return 0
    if args.plan_only:
        from collections import Counter

        print(f"{len(plan)} items")
        print(Counter((i.tier, i.family) for i in plan))
        return 0
    try:
        with single_instance(f"{JOB}_{args.shard_base + args.shard}"):
            state = run_sweep_queue(
                FMPProvider(),
                tag=args.tag,
                end=args.end,
                plan=plan,
                hours=args.hours,
                n_workers=args.workers,
                retry_failures=args.retry_failures,
                shard=args.shard,
                n_shards=args.shards,
                shard_base=args.shard_base,
            )
    except AlreadyRunning as exc:
        log.warning("%s", exc)
        return 2
    write_summary(
        JOB,
        {
            "tag": args.tag,
            "planned": len(plan),
            "done": len(state.done),
            "failed": len(state.failures),
            "leaderboard": str(state.root / "leaderboard.md"),
        },
    )
    print(f"leaderboard -> {state.root / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
