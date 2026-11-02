"""Job: refresh the price cache + universe snapshots.

python -m ehf.server.research.jobs.refresh_data
python -m ehf.server.research.jobs.refresh_data --universes etf_core --timeframes 1day,1hour
python -m ehf.server.research.jobs.refresh_data --dry-run      # count stale symbols only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from ehf.server.research.data.fmp_provider import FMPProvider
from ehf.server.research.data.refresh import DEFAULT_START, refresh
from ehf.server.research.data.universes import universe_names
from ehf.server.research.jobs import (
    AlreadyRunning,
    configure_logging,
    load_dotenv_quietly,
    single_instance,
    write_summary,
)

JOB = "refresh_data"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EHF nightly data refresh")
    p.add_argument(
        "--universes",
        default="etf_core,us_liquid",
        help=f"comma-separated; known: {', '.join(universe_names())}",
    )
    p.add_argument("--timeframes", default="1day", help="comma-separated: 1day,1hour,1min")
    p.add_argument("--start", default=DEFAULT_START, help="history start for first-time fetches")
    p.add_argument("--through", help="YYYY-MM-DD session to be current through (default: last)")
    p.add_argument("--batch", type=int, default=100, help="symbols per vendor batch")
    p.add_argument("--dry-run", action="store_true", help="resolve + count, no fetch/snapshot")
    p.add_argument(
        "--backfill", action="store_true", help="re-fetch full history from --start for everyone"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    load_dotenv_quietly()
    log = configure_logging(JOB, verbose=args.verbose)
    try:
        with single_instance(JOB):
            report = refresh(
                FMPProvider(),
                universes=tuple(s.strip() for s in args.universes.split(",") if s.strip()),
                timeframes=tuple(s.strip() for s in args.timeframes.split(",") if s.strip()),
                start=args.start,
                through=date.fromisoformat(args.through) if args.through else None,
                batch_size=args.batch,
                dry_run=args.dry_run,
                backfill=args.backfill,
            )
    except AlreadyRunning as exc:
        log.warning("%s", exc)
        return 2
    summary = report.as_dict()
    path = write_summary(JOB, summary)
    log.info("done in %.0fs; summary -> %s", report.seconds, path)
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_symbols"}, indent=1))
    return 1 if any(report.failed.values()) and not report.refreshed else 0


if __name__ == "__main__":
    sys.exit(main())
