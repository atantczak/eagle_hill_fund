#!/usr/bin/env python3
"""The Mac Studio's scheduler: EHF jobs as launchd agents, from one table.

    python infra/studio/jobs.py render            # write plists to infra/studio/launchd/
    python infra/studio/jobs.py install           # render + load into launchd (user agents)
    python infra/studio/jobs.py uninstall         # unload + remove the plists
    python infra/studio/jobs.py status            # loaded? last exit code? last summary?
    python infra/studio/jobs.py run <job>         # run one job now, in the foreground

Every job is ``nice``d so Benefit Cloud work never feels it, logs to
``var/log/<job>.launchd.log`` (launchd's stdout/stderr) and the job's
own rotating log, and runs from the repo root with the project venv. launchd
re-fires a missed calendar slot when the machine wakes, so the Studio sleeping
through a night costs one late run, not a lost one.

Adding a job = adding one ``Job`` to ``JOBS``. The plists are generated (git-ignored);
this table is the source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv311" / "bin" / "python"
VAR = ROOT / "var"
LOG_DIR = VAR / "log"
RUN_DIR = VAR / "run"
RENDER_DIR = Path(__file__).resolve().parent / "launchd"
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LABEL_PREFIX = "com.eaglehill"
NICE = 10


@dataclass(frozen=True)
class Job:
    name: str  # also the launchd label suffix and the log/summary name
    module: str  # python -m <module>
    schedule: tuple[dict[str, int], ...]  # launchd StartCalendarInterval entries
    args: tuple[str, ...] = ()
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{LABEL_PREFIX}.{self.name}"

    @property
    def plist_path(self) -> Path:
        return RENDER_DIR / f"{self.label}.plist"

    @property
    def installed_path(self) -> Path:
        return AGENTS_DIR / f"{self.label}.plist"

    def program_arguments(self) -> list[str]:
        return ["/usr/bin/nice", "-n", str(NICE), str(PYTHON), "-m", self.module, *self.args]

    def plist(self) -> dict[str, object]:
        return {
            "Label": self.label,
            "ProgramArguments": self.program_arguments(),
            "WorkingDirectory": str(ROOT),
            "StartCalendarInterval": list(self.schedule),
            "StandardOutPath": str(LOG_DIR / f"{self.name}.launchd.log"),
            "StandardErrorPath": str(LOG_DIR / f"{self.name}.launchd.log"),
            "EnvironmentVariables": {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
                "HOME": str(Path.home()),
                "LANG": "en_US.UTF-8",
                "PYTHONUNBUFFERED": "1",
                # Research on the Studio logs to the local SQLite registry; RDS is a
                # promotion target, never on a nightly sweep's write path (ADR-0002).
                "EHF_RESULTS_BACKEND": "local",
                **self.env,
            },
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "RunAtLoad": False,
            "ExitTimeOut": 3600,
        }


# ---------------------------------------------------------------------------------------
# The schedule. Times are the Studio's local clock (America/New_York).
# ---------------------------------------------------------------------------------------

JOBS: tuple[Job, ...] = (
    Job(
        name="refresh_data",
        module="ehf.research.jobs.refresh_data",
        args=("--universes", "etf_core,us_liquid,sp500,vix_family", "--timeframes", "1day"),
        schedule=({"Hour": 2, "Minute": 0},),  # every night; weekend runs find nothing stale
        description="Snapshot universes + bring the daily price cache up to the last session",
    ),
    Job(
        name="backfill_data",
        module="ehf.research.jobs.refresh_data",
        args=("--universes", "etf_core,us_liquid,sp500", "--timeframes", "1day", "--backfill"),
        schedule=({"Day": 1, "Hour": 3, "Minute": 0},),  # 1st of the month
        description="Monthly full-history re-fetch so back-adjusted prices stay current",
    ),
    Job(
        name="refresh_hourly",
        module="ehf.research.jobs.refresh_data",
        args=("--universes", "etf_core", "--timeframes", "1hour", "--start", "2023-01-01"),
        schedule=({"Weekday": 6, "Hour": 3, "Minute": 0},),  # Saturdays
        description="Weekly hourly-bar top-up for the ETF set (intraday studies)",
    ),
    Job(
        name="vix_study",
        module="ehf.research.jobs.vix_study",
        args=("--end", "2099-12-31"),  # through the last cached session
        schedule=({"Weekday": 0, "Hour": 4, "Minute": 0},),  # Sundays, after the nightly refresh
        description=(
            "Weekly VIX overlay study: nine rules + vote vs hold / 200dma, walk-forward + gauntlet"
        ),
    ),
    Job(
        name="leverage_study",
        module="ehf.research.jobs.leverage_study",
        args=("--end", "2099-12-31"),
        schedule=({"Day": 1, "Hour": 4, "Minute": 30},),  # monthly, 1st, after the backfill
        description=(
            "Monthly leverage study: 3x fund when the market turns bullish vs the plain fund, "
            "eight pairs, walk-forward + gauntlet + event study"
        ),
    ),
    Job(
        name="sync_var",
        module="ehf.research.jobs.sync_var",
        schedule=({"Hour": 6, "Minute": 0},),  # after the refresh (2:00) and Sunday's study
        description="Copy var/ (results, scorecards, daily cache) up to S3 for the deployed lab",
    ),
)


def _by_name(name: str) -> Job:
    for j in JOBS:
        if j.name == name or j.label == name:
            return j
    raise SystemExit(f"unknown job {name!r}; known: {[j.name for j in JOBS]}")


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=check)


def render() -> list[Path]:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for job in JOBS:
        with job.plist_path.open("wb") as fh:
            plistlib.dump(job.plist(), fh, sort_keys=True)
        out.append(job.plist_path)
    return out


def install() -> int:
    if not PYTHON.exists():
        raise SystemExit(f"venv python missing: {PYTHON}")
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in render():
        job = _by_name(path.stem)
        if job.installed_path.exists():
            _launchctl("bootout", f"{_domain()}/{job.label}")
        job.installed_path.write_bytes(path.read_bytes())
        res = _launchctl("bootstrap", _domain(), str(job.installed_path))
        state = "loaded" if res.returncode == 0 else f"FAILED: {res.stderr.strip()}"
        when = ", ".join(_fmt_interval(i) for i in job.schedule)
        print(f"{job.label:40s} {state:10s} {when}")
    return 0


def uninstall() -> int:
    for job in JOBS:
        if job.installed_path.exists():
            _launchctl("bootout", f"{_domain()}/{job.label}")
            job.installed_path.unlink()
            print(f"{job.label}: removed")
    return 0


def status() -> int:
    for job in JOBS:
        res = _launchctl("print", f"{_domain()}/{job.label}")
        loaded = res.returncode == 0
        last_exit = _grab(res.stdout, "last exit code") if loaded else "-"
        summary = RUN_DIR / f"{job.name}.last.json"
        last = "-"
        if summary.exists():
            try:
                data = json.loads(summary.read_text())
                last = f"{data.get('finished_at', '?')[:16]} {_brief(data)}"
            except ValueError:
                last = "unreadable summary"
        when = ", ".join(_fmt_interval(i) for i in job.schedule)
        print(f"{job.label}")
        print(f"    loaded: {loaded}   last exit: {last_exit}   schedule: {when}")
        print(f"    last run: {last}")
        print(f"    {job.description}")
    return 0


def run(name: str) -> int:
    job = _by_name(name)
    print("+", " ".join(job.program_arguments()))
    return subprocess.call(job.program_arguments(), cwd=ROOT, env={**os.environ, **job.env})


def _grab(text: str, key: str) -> str:
    for line in text.splitlines():
        if key in line:
            return line.split("=")[-1].strip()
    return "?"


def _brief(data: dict[str, object]) -> str:
    keys = ("symbols", "refreshed", "failed", "seconds", "status")
    return " ".join(f"{k}={data[k]}" for k in keys if k in data)


def _fmt_interval(i: dict[str, int]) -> str:
    days = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    if "Day" in i:
        day = f"monthly d{i['Day']} "
    else:
        day = days.get(i["Weekday"], "?") + " " if "Weekday" in i else "daily "
    return f"{day}{i.get('Hour', 0):02d}:{i.get('Minute', 0):02d}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("render", "install", "uninstall", "status"):
        sub.add_parser(c)
    r = sub.add_parser("run")
    r.add_argument("job")
    a = p.parse_args()
    if a.cmd == "render":
        for path in render():
            print(path)
        return 0
    if a.cmd == "run":
        return run(a.job)
    return {"install": install, "uninstall": uninstall, "status": status}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
