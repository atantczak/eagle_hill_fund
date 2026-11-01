"""The daily job: snapshot -> narrative -> render -> store -> email. Also the CLI.

    python -m ehf.server.research.reporting.job --dry-run --out /tmp/report
    python -m ehf.server.research.reporting.job --max-symbols 150 --no-narrative --dry-run

``--dry-run`` skips S3 and email but still writes every artifact to ``--out``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.deliver import publish_s3, send_email
from ehf.server.research.reporting.fmp import FMPReportClient
from ehf.server.research.reporting.narrative import write_narrative
from ehf.server.research.reporting.render import render_html, render_markdown
from ehf.server.research.reporting.snapshot import Snapshot, build_snapshot

log = logging.getLogger(__name__)


@dataclass
class RunOutput:
    snapshot: Snapshot
    narrative: str | None
    markdown: str
    html: str
    artifacts: dict[str, bytes]
    s3_objects: list[str] = field(default_factory=list)
    email_id: str | None = None
    local_paths: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "run_date": self.snapshot.run_date.isoformat(),
            "data_as_of": self.snapshot.as_of.isoformat(),
            "universe": int(len(self.snapshot.universe)),
            "hits": {k: len(v) for k, v in self.snapshot.hits.items()},
            "narrative": self.narrative is not None,
            "api_calls": self.snapshot.api_calls,
            "timings": self.snapshot.timings,
            "warnings": self.snapshot.warnings,
            "s3_objects": self.s3_objects,
            "email_id": self.email_id,
            "local_paths": self.local_paths,
        }


def build_artifacts(
    snapshot: Snapshot, narrative: str | None, markdown: str, html: str, cfg: ReportConfig
) -> dict[str, bytes]:
    """Everything worth keeping from a run, as ``filename -> bytes``."""
    hits = {
        "run_date": snapshot.run_date.isoformat(),
        "data_as_of": snapshot.as_of.isoformat(),
        "hits": snapshot.hits_json(),
    }
    meta = {
        "run_date": snapshot.run_date.isoformat(),
        "data_as_of": snapshot.as_of.isoformat(),
        "universe_size": int(len(snapshot.universe)),
        "n_hits": snapshot.n_hits,
        "api_calls": snapshot.api_calls,
        "timings": snapshot.timings,
        "warnings": snapshot.warnings,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "narrative_model": cfg.anthropic_model if narrative else None,
    }
    table_csv = snapshot.table().to_csv(index=True).encode()
    out = {
        "report.md": markdown.encode(),
        "report.html": html.encode(),
        "hits.json": json.dumps(hits, indent=1, default=str).encode(),
        "market.json": json.dumps(snapshot.market, indent=1, default=str).encode(),
        "universe_table.csv.gz": gzip.compress(table_csv),
        "meta.json": json.dumps(meta, indent=1, default=str).encode(),
    }
    if narrative:
        out["narrative.md"] = narrative.encode()
    return out


def run(
    cfg: ReportConfig,
    *,
    run_date: date | None = None,
    dry_run: bool = False,
    out_dir: str | Path | None = None,
    fmp_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> RunOutput:
    t0 = time.perf_counter()
    key = fmp_api_key or os.getenv("FMP_API_KEY") or ""
    client = FMPReportClient(key, workers=cfg.fetch_workers)
    snapshot = build_snapshot(client, cfg, run_date=run_date)
    log.info(
        "snapshot: %d symbols, %d hits, %d API calls",
        len(snapshot.universe),
        snapshot.n_hits,
        snapshot.api_calls,
    )

    narrative: str | None = None
    if cfg.narrative:
        t1 = time.perf_counter()
        narrative = write_narrative(
            snapshot.narrative_payload(),
            cfg,
            api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
        snapshot.timings["narrative"] = round(time.perf_counter() - t1, 2)
        if narrative is None:
            snapshot.warnings.append("narrative unavailable (see logs)")

    markdown = render_markdown(snapshot, narrative)
    html = render_html(snapshot, narrative)
    artifacts = build_artifacts(snapshot, narrative, markdown, html, cfg)
    out = RunOutput(
        snapshot=snapshot, narrative=narrative, markdown=markdown, html=html, artifacts=artifacts
    )

    if out_dir is not None:
        d = Path(out_dir) / snapshot.run_date.isoformat()
        d.mkdir(parents=True, exist_ok=True)
        for name, blob in artifacts.items():
            p = d / name
            p.write_bytes(blob)
            out.local_paths.append(str(p))

    if not dry_run:
        if cfg.s3_bucket:
            out.s3_objects = publish_s3(
                cfg.s3_bucket, cfg.s3_prefix, snapshot.run_date, artifacts, region=cfg.aws_region
            )
        if cfg.email_from and cfg.email_to:
            regime = snapshot.market.get("index", {}).get("regime", "regime n/a")
            subject = f"EHF daily report {snapshot.run_date} — {snapshot.n_hits} hits, {regime}"
            out.email_id = send_email(
                sender=cfg.email_from,
                recipients=list(cfg.email_to),
                subject=subject,
                html_body=html,
                text_body=markdown,
                region=cfg.aws_region,
            )
    snapshot.timings["total"] = round(time.perf_counter() - t0, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eagle Hill daily market report")
    p.add_argument("--date", help="run date YYYY-MM-DD (default: today)")
    p.add_argument("--dry-run", action="store_true", help="skip S3 + email")
    p.add_argument("--out", help="directory to write artifacts into (a dated subfolder is created)")
    p.add_argument(
        "--max-symbols", type=int, help="cap the universe (largest caps first) for quick runs"
    )
    p.add_argument("--no-narrative", action="store_true", help="skip the Claude narrative")
    p.add_argument("--workers", type=int, help="FMP fetch concurrency")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = ReportConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.max_symbols:
        overrides["max_symbols"] = args.max_symbols
    if args.no_narrative:
        overrides["narrative"] = False
    if args.workers:
        overrides["fetch_workers"] = args.workers
    if overrides:
        cfg = ReportConfig(**{**cfg.__dict__, **overrides})

    run_date = date.fromisoformat(args.date) if args.date else None
    out = run(cfg, run_date=run_date, dry_run=args.dry_run, out_dir=args.out)
    print(json.dumps(out.summary(), indent=1, default=str))
    if not args.out:
        print(out.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
