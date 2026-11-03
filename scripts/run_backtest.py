"""Run a single backtest from the command line — pick the strategy, universe,
dates, and params; print the metrics; log it to the results store (RDS+S3 when
configured, else local).

This is the thin CLI around the three-step recipe:

    spec  = RunSpec(...)                 # the reproducible description of a run
    bars  = FMPProvider().get_bars(...)  # cached OHLCV for the universe + dates
    res   = run_backtest(spec, bars)     # equity curve + trade ledger + metrics
    ExperimentRegistry().log_run(...)    # persist (idempotent, hashed by config)

Examples
--------
  # Time-series momentum, vol-scaled, on a few names over 6 years:
  python scripts/run_backtest.py \
      --strategy tsmom --symbols AAPL,MSFT,NVDA,AMZN,GOOGL \
      --start 2018-01-01 --end 2024-12-31

  # Regime-gated momentum with overridden params, longer slippage:
  python scripts/run_backtest.py \
      --strategy tsmom_regime --symbols AAPL,MSFT,NVDA \
      --start 2015-01-01 --end 2024-12-31 \
      --param lookback=126 --param hurst_min=0.45 --slippage-bps 2

  # Don't persist (just look at the numbers):
  python scripts/run_backtest.py --strategy rise_fall \
      --symbols AAPL --start 2020-01-01 --end 2024-12-31 --no-log

Run ``--help`` for the full flag list. The set of strategies and their tunable
params is whatever is registered (see ``--list``).
"""

from __future__ import annotations

import argparse

from ehf.research import strategies  # noqa: F401  (import registers strategies)
from ehf.research.backtest.engine import run_backtest
from ehf.research.config.spec import CostConfig, FillModel, RunSpec, UniverseSpec
from ehf.research.data.fmp_provider import FMPProvider
from ehf.research.results import ExperimentRegistry
from ehf.research.strategies.base import _REGISTRY

# Order the metrics print so the "is this any good?" numbers come first.
_METRIC_ORDER = [
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "ann_vol",
    "total_return",
    "hit_rate",
    "n_trades",
    "avg_turnover",
    "n_bars",
]
_PCT_METRICS = {"cagr", "max_drawdown", "ann_vol", "total_return", "hit_rate"}


def _coerce(value: str) -> object:
    """Turn a ``--param k=v`` string into int/float/bool/str (in that order)."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = _coerce(raw.strip())
    return out


def _print_strategy_menu() -> None:
    print("Registered strategies and their tunable params (override with --param k=v):\n")
    for key in sorted(_REGISTRY):
        cls = _REGISTRY[key]
        params = ", ".join(f"{p}={v}" for p, v in cls.defaults.items())
        print(f"  {key:20} {params}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="list strategies + params, then exit")
    ap.add_argument("--strategy", help="registered strategy key (see --list)")
    ap.add_argument("--symbols", help="comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    ap.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--timeframe", default="1day", help="bar size (default: 1day)")
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="override a strategy param (repeatable); types are inferred",
    )
    ap.add_argument("--slippage-bps", type=float, default=1.0, help="per-trade slippage (bps)")
    ap.add_argument("--initial-capital", type=float, default=1_000_000.0)
    ap.add_argument(
        "--fill",
        choices=[f.value for f in FillModel],
        default=FillModel.NEXT_OPEN.value,
        help="fill model (default: next_open, the safe one)",
    )
    ap.add_argument("--no-log", action="store_true", help="run but do NOT persist to the store")
    args = ap.parse_args()

    if args.list:
        _print_strategy_menu()
        return

    missing = [f for f in ("strategy", "symbols", "start", "end") if not getattr(args, f)]
    if missing:
        flags = ", ".join("--" + m for m in missing)
        raise SystemExit(f"missing required args: {flags} (or use --list)")
    if args.strategy not in _REGISTRY:
        raise SystemExit(f"unknown strategy {args.strategy!r}; choices: {sorted(_REGISTRY)}")

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())

    # 1. The reproducible description of the run (hashed -> results-store key).
    spec = RunSpec(
        strategy=args.strategy,
        params=_parse_params(args.param),
        universe=UniverseSpec(symbols=symbols),
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        initial_capital=args.initial_capital,
        costs=CostConfig(slippage_bps=args.slippage_bps),
        fill_model=FillModel(args.fill),
    )

    # 2. Fetch OHLCV (transparently cached to parquet; re-runs hit cache, not the API).
    print(f"fetching {len(symbols)} symbols {args.start}..{args.end} ...")
    bars = FMPProvider().get_bars(list(symbols), args.start, args.end, args.timeframe)
    print(f"  {len(bars)} bars across {bars['symbol'].nunique()} symbols\n")

    # 3. Run the backtest (look-ahead-safe, cost-aware engine).
    result = run_backtest(spec, bars)

    print(f"=== {args.strategy} | {','.join(symbols)} | {args.start}..{args.end} ===")
    print(f"config_hash: {spec.config_hash}")
    for name in _METRIC_ORDER:
        if name not in result.metrics:
            continue
        v = result.metrics[name]
        shown = f"{v:>10.2%}" if name in _PCT_METRICS else f"{v:>10.2f}"
        print(f"  {name:14} {shown}")

    # 4. Persist (unless --no-log). Idempotent: same config_hash won't double-write.
    if args.no_log:
        print("\n(not logged: --no-log)")
        return
    registry = ExperimentRegistry()
    backend = type(registry.store).__name__
    registry.log_run(spec, result)
    print(f"\nlogged to {backend} (config_hash={spec.config_hash})")


if __name__ == "__main__":
    main()
