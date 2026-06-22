"""Benchmark: serial vs parallel parameter sweep on real data. Run: python scripts/bench_sweep.py"""
import warnings, logging, os, time, tempfile
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from ehf.server.research.data.fmp_provider import FMPProvider
from ehf.server.research.data.schema import Timeframe
from ehf.server.research.config.spec import RunSpec, UniverseSpec, CostConfig
from ehf.server.research.backtest.engine import run_backtest
from ehf.server.research.backtest.sweep import run_sweep, expand_grid, make_sweep_id
from ehf.server.research.results.registry import ExperimentRegistry
from ehf.server.research.results.store import LocalResultStore

def main():
    SYMBOLS = ["AAPL","MSFT","JPM","XOM","KO","JNJ","WMT","PG","DIS","VZ",
               "NVDA","AMZN","GOOGL","META","HD","BAC","CVX","PFE","CSCO","INTC",
               "MRK","ABT","CRM","ORCL"]
    bars = FMPProvider().get_bars(SYMBOLS, "2010-01-01","2024-12-31", Timeframe.DAY)
    base = RunSpec(strategy="rise_fall", params={"atr_window":14},
        universe=UniverseSpec(symbols=tuple(SYMBOLS)),
        start="2010-01-01", end="2024-12-31", costs=CostConfig(slippage_bps=1.0))
    grid = {"trend_window":[50,100,150,200], "drop_pct":[0.003,0.005,0.01,0.02],
            "rise_pct":[0.02,0.04], "max_hold":[10,20]}
    specs = expand_grid(base, grid, make_sweep_id(base, grid))
    print(f"data: {len(bars)} rows, {bars['symbol'].nunique()} symbols | "
          f"cores: {os.cpu_count()} | cells: {len(specs)}\n")

    t0=time.perf_counter()
    for s in specs: run_backtest(s, bars)
    serial=time.perf_counter()-t0
    print(f"SERIAL  (1 core)    : {serial:6.2f}s  ({serial/len(specs)*1000:.0f} ms/cell)")

    reg = ExperimentRegistry(store=LocalResultStore(root=tempfile.mkdtemp()))
    t0=time.perf_counter()
    out = run_sweep(base, grid, bars, registry=reg)
    par=time.perf_counter()-t0
    print(f"PARALLEL ({os.cpu_count()} cores) : {par:6.2f}s  ({par/len(specs)*1000:.0f} ms/cell)")
    print(f"\n>>> SPEEDUP: {serial/par:.1f}x   (cells logged: {len(out)})")

if __name__ == "__main__":
    main()
