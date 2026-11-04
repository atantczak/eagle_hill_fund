"""Alternative data from FMP, made backtestable: fundamentals, analysts, earnings,
insiders, Congress, rates, jobless claims, copper/gold, the dollar.

The engine only sees bars, so every non-price series rides along as extra
**feature columns**: pseudo-symbols named ``^<FEAT>.<SYM>`` (per stock) or
``^<NAME>`` (macro), with open = high = low = close = the value and zero volume.
Strategies read them from ``panel.close`` and never put weight on them (the same
trick the VIX study uses for ``^VIX``).

Point-in-time discipline, the whole point of this module:

* every value is stamped on the first session **strictly after** the day the
  market could see it: an earnings announcement date, a filing's ``filingDate``,
  a Congress ``disclosureDate`` (not the trade date), a ratings change date;
* macro series are lagged past their release (claims: week end + 7 days;
  treasury yields: next session);
* fundamentals use filing dates, never fiscal period ends;
* S&P 500 membership is rebuilt month by month from the change log
  (``universes.sp500_members_as_of``) and carried as a ``^MEM.<SYM>`` column.

Known limits, stated in every write-up: names that left the index and later
vanished from FMP have no bars (mild survivorship flattery); FMP's economic series
are today's revised values, not first prints.

Step series are stored sparsely (at filings, month starts or events; strategies
forward-fill), so a dataset of ~800 stocks stays a few million rows. Anything a
strategy forward-fills must recur more often than the walk-forward warm-up (252
sessions), or a sliced window would lose it: membership is stamped monthly. Raw JSON is cached per
(dataset, symbol) under ``var/cache/fmp_alt``; datasets are cached as parquet.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ehf.research.data.schema import BARS_COLUMNS, Timeframe, validate_bars
from ehf.research.data.universes import sp500_members_as_of
from ehf.research.data.vix import cash_bars_from_yield
from ehf.research.paths import CACHE_DIR

log = logging.getLogger(__name__)

ALT_DIR = CACHE_DIR / "fmp_alt"
DATA_START = "2009-01-01"
STOCK_START = "2010-01-01"  # first membership month for the stock datasets

STOCK_DATASETS = ("sp500_pead", "sp500_grades", "sp500_fund", "sp500_insider", "sp500_congress")
MACRO_DATASETS = ("macro_SPY", "macro_QQQ", "macro_IWM")
ROTATION_DATASETS = ("cugo_sectors", "dollar_em")
ALL_DATASETS = STOCK_DATASETS + MACRO_DATASETS + ROTATION_DATASETS

CYCLICALS = ("XLB", "XLI", "XLE", "XLF", "XLY", "XLK")
DEFENSIVES = ("XLU", "XLP", "XLV")


def feat(name: str, sym: str | None = None) -> str:
    return f"^{name}.{sym}" if sym else f"^{name}"


# --- transport + raw cache --------------------------------------------------------------


def _client():  # type: ignore[no-untyped-def]
    from ehf.research.reporting.fmp import FMPReportClient

    return FMPReportClient(os.environ["FMP_API_KEY"], workers=16)


def _cached_json(kind: str, key: str, fetch) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    path = ALT_DIR / kind / f"{key.replace('/', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        rows = fetch()
    except Exception as exc:  # leave it uncached so a later run retries
        log.warning("fmp %s %s: %s", kind, key, exc)
        return []
    rows = rows if isinstance(rows, list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)
    return rows


def _paged(c, endpoint: str, *, max_pages: int, stop_before: str, date_key: str, **kw):  # type: ignore[no-untyped-def]
    out: list[dict[str, Any]] = []
    for page in range(max_pages):
        rows = c.get_json(endpoint, page=page, limit=100, **kw) or []
        out += rows
        if len(rows) < 100 or min(str(r.get(date_key) or "9999") for r in rows) < stop_before:
            break
    return out


def fetch_symbol_data(symbols: Iterable[str], kinds: Iterable[str]) -> None:
    """Warm the raw cache for every (kind, symbol) in parallel."""
    c = _client()
    kinds = list(kinds)
    fetchers = {
        "earnings": lambda s: c.get_json("earnings", symbol=s, limit=1000),
        "grades": lambda s: c.get_json("grades", symbol=s),
        "income": lambda s: c.get_json("income-statement", symbol=s, period="quarter", limit=80),
        "balance": lambda s: c.get_json(
            "balance-sheet-statement", symbol=s, period="quarter", limit=80
        ),
        "cashflow": lambda s: c.get_json(
            "cash-flow-statement", symbol=s, period="quarter", limit=80
        ),
        "mktcap": lambda s: _mktcap(c, s),
        "insider": lambda s: _paged(
            c,
            "insider-trading/search",
            symbol=s,
            max_pages=40,
            stop_before=STOCK_START,
            date_key="filingDate",
        ),
        "senate": lambda s: _paged(
            c,
            "senate-trades",
            symbol=s,
            max_pages=20,
            stop_before="2012-01-01",
            date_key="disclosureDate",
        ),
        "house": lambda s: _paged(
            c,
            "house-trades",
            symbol=s,
            max_pages=20,
            stop_before="2012-01-01",
            date_key="disclosureDate",
        ),
    }
    tasks = [(k, s) for k in kinds for s in symbols if not (ALT_DIR / k / f"{s}.json").exists()]
    log.info("fetching %d (kind, symbol) pairs", len(tasks))

    def one(t: tuple[str, str]) -> None:
        k, s = t
        _cached_json(k, s, lambda: fetchers[k](s))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, tasks))


def _mktcap(c, sym: str) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    out: list[dict[str, Any]] = []
    for lo, hi in (("2008-06-01", "2017-12-31"), ("2018-01-01", date.today().isoformat())):
        out += (
            c.get_json(
                "historical-market-capitalization", symbol=sym, limit=5000, **{"from": lo, "to": hi}
            )
            or []
        )
    return out


def _raw(kind: str, sym: str) -> list[dict[str, Any]]:
    path = ALT_DIR / kind / f"{sym}.json"
    return json.loads(path.read_text()) if path.exists() else []


# --- stamping helpers (pure) ------------------------------------------------------------


def next_session(sessions: pd.DatetimeIndex, when: Iterable[Any], lag_days: int = 0) -> np.ndarray:
    """Index of the first session strictly after each date (+ ``lag_days`` calendar days)."""
    ts = pd.to_datetime(pd.Series(list(when)), errors="coerce", utc=True)
    ts = ts.dt.normalize() + pd.Timedelta(days=lag_days)
    days = sessions.normalize()
    return days.searchsorted(ts.to_numpy(), side="right")


def feature_bars(
    symbol: str,
    sessions: pd.DatetimeIndex,
    pos: np.ndarray,
    values: Iterable[float],
    *,
    how: str = "last",
) -> pd.DataFrame:
    """Feature rows on sessions[pos]; several events on one session are summed or last-wins."""
    v = pd.Series(list(values), dtype="float64")
    ok = (pos < len(sessions)) & v.notna().to_numpy()
    if not ok.any():
        return pd.DataFrame(columns=list(BARS_COLUMNS))
    s = pd.Series(v.to_numpy()[ok], index=sessions[pos[ok]])
    s = s.groupby(level=0).sum() if how == "sum" else s.groupby(level=0).last()
    return index_rows(symbol, s)


def index_rows(symbol: str, s: pd.Series) -> pd.DataFrame:
    s = s.dropna().astype("float64")
    return pd.DataFrame(
        {
            "timestamp": s.index,
            "symbol": symbol,
            "open": s.to_numpy(),
            "high": s.to_numpy(),
            "low": s.to_numpy(),
            "close": s.to_numpy(),
            "volume": 0.0,
        }
    )


def change_points(s: pd.Series) -> pd.Series:
    """Keep only the rows where a step series changes (strategies forward-fill)."""
    s = s.dropna()
    return s[s.ne(s.shift())]


# --- per-stock features (pure, from raw JSON rows) -------------------------------------


def surprise_feature(
    rows: list[dict[str, Any]], sym: str, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Earnings surprise (actual - estimate) / max(|estimate|, 0.05), clipped to +-1,
    stamped on the session after the announcement date."""
    rows = [r for r in rows if r.get("epsActual") is not None and r.get("epsEstimated") is not None]
    if not rows:
        return pd.DataFrame(columns=list(BARS_COLUMNS))
    a = np.array([float(r["epsActual"]) for r in rows])
    e = np.array([float(r["epsEstimated"]) for r in rows])
    sue = np.clip((a - e) / np.maximum(np.abs(e), 0.05), -1.0, 1.0)
    pos = next_session(sessions, [r["date"] for r in rows])
    return feature_bars(feat("SUE", sym), sessions, pos, sue)


def grade_feature(rows: list[dict[str, Any]], sym: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Net analyst actions per session: +1 per upgrade, -1 per downgrade."""
    score = {"upgrade": 1.0, "downgrade": -1.0}
    rows = [r for r in rows if str(r.get("action", "")).lower() in score]
    pos = next_session(sessions, [r["date"] for r in rows])
    vals = [score[str(r["action"]).lower()] for r in rows]
    return feature_bars(feat("GRD", sym), sessions, pos, vals, how="sum")


def insider_feature(
    rows: list[dict[str, Any]], sym: str, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Open-market purchases (Form 4 code P) per session, by filing date; one per insider."""
    buys = [r for r in rows if str(r.get("transactionType", "")).startswith("P")]
    seen: set[tuple[str, str]] = set()
    keep = []
    for r in buys:
        k = (
            str(r.get("filingDate", ""))[:10],
            str(r.get("reportingCik") or r.get("reportingName")),
        )
        if k not in seen:
            seen.add(k)
            keep.append(r)
    pos = next_session(sessions, [str(r.get("filingDate", ""))[:10] for r in keep])
    return feature_bars(feat("INS", sym), sessions, pos, [1.0] * len(keep), how="sum")


def congress_feature(
    rows: list[dict[str, Any]], sym: str, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Net Congress trades per session by disclosure date: +1 purchase, -1 sale."""
    vals, dates = [], []
    for r in rows:
        t = str(r.get("type", "")).lower()
        if "purchase" in t:
            vals.append(1.0)
        elif "sale" in t:
            vals.append(-1.0)
        else:
            continue
        dates.append(r.get("disclosureDate"))
    pos = next_session(sessions, dates)
    return feature_bars(feat("CNG", sym), sessions, pos, vals, how="sum")


def _num(r: dict[str, Any], *keys: str) -> float:
    for k in keys:
        v = r.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float("nan")


def fundamentals_table(sym: str) -> pd.DataFrame:
    """Quarterly table (by fiscal period end) with TTM flows, F-score inputs and the
    date each quarter became public (the latest filing date of its three statements)."""
    inc = {r["date"]: r for r in _raw("income", sym)}
    bal = {r["date"]: r for r in _raw("balance", sym)}
    cfs = {r["date"]: r for r in _raw("cashflow", sym)}
    dates = sorted(set(inc) & set(bal) & set(cfs))
    if len(dates) < 9:
        return pd.DataFrame()
    recs = []
    for d in dates:
        i, b, c = inc[d], bal[d], cfs[d]
        filed = max(str(x.get("filingDate") or "") for x in (i, b, c)) or None
        if not filed:
            filed = (pd.Timestamp(d) + pd.Timedelta(days=60)).date().isoformat()
        recs.append(
            {
                "period": d,
                "filed": filed,
                "ni": _num(i, "netIncome"),
                "rev": _num(i, "revenue"),
                "gp": _num(i, "grossProfit"),
                "shares": _num(i, "weightedAverageShsOutDil", "weightedAverageShsOut"),
                "assets": _num(b, "totalAssets"),
                "ltd": _num(b, "longTermDebt"),
                "ca": _num(b, "totalCurrentAssets"),
                "cl": _num(b, "totalCurrentLiabilities"),
                "cfo": _num(c, "operatingCashFlow", "netCashProvidedByOperatingActivities"),
                "div": _num(c, "commonDividendsPaid", "netDividendsPaid", "dividendsPaid"),
                "netiss": _num(c, "netCommonStockIssuance"),
                "rep": _num(c, "commonStockRepurchased"),
                "iss": _num(c, "commonStockIssuance", "commonStockIssued"),
            }
        )
    q = pd.DataFrame(recs).sort_values("period").reset_index(drop=True)
    for col in ("ni", "rev", "gp", "cfo", "div", "netiss", "rep", "iss"):
        q[f"{col}_ttm"] = q[col].rolling(4, min_periods=4).sum()
    buyback = q["netiss_ttm"].where(
        q["netiss_ttm"].notna(), q["rep_ttm"].fillna(0) + q["iss_ttm"].fillna(0)
    )
    q["payout_ttm"] = -q["div_ttm"].fillna(0.0) - buyback.fillna(0.0)
    roa = q["ni_ttm"] / q["assets"]
    lev = q["ltd"].fillna(0.0) / q["assets"]
    cur = q["ca"] / q["cl"]
    gm = q["gp_ttm"] / q["rev_ttm"]
    turn = q["rev_ttm"] / q["assets"].shift(4)
    lag = 4
    pts = [
        roa > 0,
        q["cfo_ttm"] > 0,
        roa > roa.shift(lag),
        q["cfo_ttm"] > q["ni_ttm"],
        lev < lev.shift(lag),
        cur > cur.shift(lag),
        q["shares"] <= q["shares"].shift(lag),
        gm > gm.shift(lag),
        turn > turn.shift(lag),
    ]
    q["fscore"] = sum(p.astype(float) for p in pts)
    q.loc[q.index < 8, "fscore"] = np.nan  # needs two full years of quarters
    q["filed"] = pd.to_datetime(q["filed"]).dt.strftime("%Y-%m-%d")
    # a later quarter can't be public before an earlier one: enforce monotone availability
    q["filed"] = q["filed"].cummax()
    return q


def fundamental_features(sym: str, sessions: pd.DatetimeIndex) -> list[pd.DataFrame]:
    """F-score (step, at filing), earnings yield and payout yield (month-end samples
    of TTM flow / market cap, using only quarters filed by then)."""
    q = fundamentals_table(sym)
    if q.empty:
        return []
    out = []
    pos = next_session(sessions, q["filed"])
    fs = feature_bars(feat("FS", sym), sessions, pos, q["fscore"])
    out.append(fs)
    mc = pd.DataFrame(_raw("mktcap", sym))
    if mc.empty or "marketCap" not in mc:
        return out
    mc = mc.assign(date=pd.to_datetime(mc["date"], utc=True)).drop_duplicates("date")
    mc = mc.set_index("date")["marketCap"].astype(float).sort_index()
    month_ends = mc.groupby(mc.index.to_period("M").astype(str)).tail(1)
    known_at = pd.Series(sessions[np.minimum(pos, len(sessions) - 1)], index=q.index)
    known_at[pos >= len(sessions)] = pd.NaT
    ey, py = [], []
    for ts, cap in month_ends.items():
        avail = q[known_at.notna() & (known_at <= ts)]
        if avail.empty or not cap or not math.isfinite(cap) or cap <= 0:
            ey.append(np.nan)
            py.append(np.nan)
            continue
        last = avail.iloc[-1]
        ey.append(last["ni_ttm"] / cap)
        py.append(last["payout_ttm"] / cap)
    p2 = next_session(sessions, month_ends.index.strftime("%Y-%m-%d"))
    out.append(feature_bars(feat("EY", sym), sessions, p2, ey))
    out.append(feature_bars(feat("PY", sym), sessions, p2, py))
    return out


# --- membership --------------------------------------------------------------------------


def sp500_membership(provider, start: str, end: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Month-start S&P 500 membership as a 0/1 frame (rows = month starts, cols = symbols)."""
    current, changes = provider._sp500_current(), provider._sp500_changes()
    months = pd.date_range(start, end, freq="MS")
    sets = {m: sp500_members_as_of(current, changes, m.date()) for m in months}
    syms = sorted(set().union(*sets.values()))
    frame = pd.DataFrame(0.0, index=months, columns=syms)
    for m, members in sets.items():
        frame.loc[m, sorted(members)] = 1.0
    return frame


# --- macro series --------------------------------------------------------------------------


def treasury_curve(end: str) -> pd.DataFrame:
    c = _client()

    def fetch() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        lo = pd.Timestamp("2002-01-01")
        while lo <= pd.Timestamp(end):
            hi = lo + pd.Timedelta(days=85)
            rows += (
                c.get_json(
                    "treasury-rates", **{"from": lo.date().isoformat(), "to": hi.date().isoformat()}
                )
                or []
            )
            lo = hi + pd.Timedelta(days=1)
        return rows

    df = pd.DataFrame(_cached_json("macro", f"treasury_{end}", fetch))
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()


def initial_claims(end: str) -> pd.Series:
    c = _client()

    def fetch() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for y in range(2000, pd.Timestamp(end).year + 1):
            rows += (
                c.get_json(
                    "economic-indicators",
                    name="initialClaims",
                    **{"from": f"{y}-01-01", "to": f"{y}-12-31"},
                )
                or []
            )
        return rows

    df = pd.DataFrame(_cached_json("macro", f"claims_{end}", fetch))
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date")["value"].astype(float).sort_index()


# --- dataset assembly ---------------------------------------------------------------------


def _finish(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    pieces = [p for p in pieces if p is not None and not p.empty]
    out = pd.concat(pieces, ignore_index=True)[list(BARS_COLUMNS)]
    out["symbol"] = out["symbol"].astype("string").str.upper()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(["symbol", "timestamp"], keep="last")
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out.attrs["timeframe"] = Timeframe.DAY.value
    out.attrs["adjusted"] = True
    return validate_bars(out)


def _sessions(provider, end: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:  # type: ignore[no-untyped-def]
    spy = provider.get_bars(["SPY"], DATA_START, end, Timeframe.DAY)
    return spy, pd.DatetimeIndex(sorted(spy["timestamp"].unique()))


def build_stock_dataset(name: str, provider, end: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    member = sp500_membership(provider, STOCK_START, end)
    spy, sessions = _sessions(provider, end)
    syms = list(member.columns)
    kinds = {
        "sp500_pead": ["earnings"],
        "sp500_grades": ["grades"],
        "sp500_fund": ["income", "balance", "cashflow", "mktcap"],
        "sp500_insider": ["insider"],
        "sp500_congress": ["senate", "house"],
    }[name]
    fetch_symbol_data(syms, kinds)
    bars = _bars_tolerant(provider, syms, end)
    have = sorted(set(bars["symbol"].astype(str)))
    log.info("%s: %d of %d PIT members have bars", name, len(have), len(syms))
    pieces = [bars, spy] if "SPY" not in have else [bars]
    for sym in have:
        # every month start, not just changes: a walk-forward window slices the bars, and
        # a change point older than the slice would vanish (members would read as absent)
        m = member[sym]
        pos = next_session(sessions, m.index.strftime("%Y-%m-%d"), lag_days=-1)
        pieces.append(feature_bars(feat("MEM", sym), sessions, pos, m.to_numpy()))
        if name == "sp500_pead":
            pieces.append(surprise_feature(_raw("earnings", sym), sym, sessions))
        elif name == "sp500_grades":
            pieces.append(grade_feature(_raw("grades", sym), sym, sessions))
        elif name == "sp500_insider":
            pieces.append(insider_feature(_raw("insider", sym), sym, sessions))
        elif name == "sp500_congress":
            pieces.append(congress_feature(_raw("senate", sym) + _raw("house", sym), sym, sessions))
        elif name == "sp500_fund":
            pieces += fundamental_features(sym, sessions)
    return _finish(pieces)


def _bars_tolerant(provider, syms: list[str], end: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Bars for every symbol FMP has; skip the ones it does not (delisted, renamed)."""
    frames = []
    chunk = 50
    for i in range(0, len(syms), chunk):
        part = syms[i : i + chunk]
        try:
            frames.append(provider.get_bars(part, DATA_START, end, Timeframe.DAY))
        except Exception:
            for s in part:
                try:
                    frames.append(provider.get_bars([s], DATA_START, end, Timeframe.DAY))
                except Exception as exc:
                    log.info("no bars for %s (%s)", s, type(exc).__name__)
    got = set().union(*(set(f["symbol"].astype(str)) for f in frames if not f.empty))
    for s in [s for s in syms if s not in got]:  # a throttled batch can drop names silently
        try:
            f = provider.get_bars([s], DATA_START, end, Timeframe.DAY)
            if not f.empty:
                frames.append(f)
        except Exception as exc:
            log.info("no bars for %s (%s)", s, type(exc).__name__)
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return out[out["timestamp"] >= pd.Timestamp(DATA_START, tz="UTC")]


def build_macro_dataset(asset: str, provider, end: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    raw = provider.get_bars([asset, "^IRX"], "2003-01-01", end, Timeframe.DAY)
    a = raw[raw["symbol"] == asset]
    sessions = pd.DatetimeIndex(sorted(a["timestamp"].unique()))
    irx = (
        raw[raw["symbol"] == "^IRX"]
        .set_index("timestamp")["close"]
        .reindex(sessions)
        .ffill(limit=5)
    )
    cash = cash_bars_from_yield(irx.fillna(0.0))
    tr = treasury_curve(end)
    spread = (tr["year10"] - tr["month3"]).dropna()
    pos = next_session(sessions, spread.index.strftime("%Y-%m-%d"))
    curve = feature_bars(feat("T10Y3M"), sessions, pos, spread.to_numpy())
    claims = initial_claims(end)
    c4 = claims.rolling(4).mean().dropna()
    pos = next_session(sessions, c4.index.strftime("%Y-%m-%d"), lag_days=6)  # Thursday release
    cl = feature_bars(feat("CLAIMS4W"), sessions, pos, c4.to_numpy())
    return _finish([a, cash, curve, cl])


def build_rotation_dataset(name: str, provider, end: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    if name == "cugo_sectors":
        tradable = [*CYCLICALS, *DEFENSIVES, "SPY"]
        raw = provider.get_bars([*tradable, "HGUSD", "GCUSD"], "2003-01-01", end, Timeframe.DAY)
        bars = raw[raw["symbol"].isin(tradable)]
        sessions = pd.DatetimeIndex(sorted(bars.loc[bars["symbol"] == "SPY", "timestamp"].unique()))
        w = raw.pivot(index="timestamp", columns="symbol", values="close")
        ratio = (w["HGUSD"] / w["GCUSD"]).dropna()
        pos = next_session(sessions, ratio.index.strftime("%Y-%m-%d"), lag_days=-1)
        return _finish([bars, feature_bars(feat("CUGO"), sessions, pos, ratio.to_numpy())])
    if name == "dollar_em":
        tradable = ["SPY", "EEM", "EFA"]
        raw = provider.get_bars([*tradable, "DX-Y.NYB"], "2003-01-01", end, Timeframe.DAY)
        bars = raw[raw["symbol"].isin(tradable)]
        sessions = pd.DatetimeIndex(sorted(bars.loc[bars["symbol"] == "SPY", "timestamp"].unique()))
        dxy = raw[raw["symbol"] == "DX-Y.NYB"].set_index("timestamp")["close"].dropna()
        pos = next_session(sessions, dxy.index.strftime("%Y-%m-%d"), lag_days=-1)
        return _finish([bars, feature_bars(feat("DXY"), sessions, pos, dxy.to_numpy())])
    raise KeyError(name)


def load_dataset(name: str, provider, end: str, *, rebuild: bool = False) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    path = ALT_DIR / "datasets" / f"{name}_{end}.parquet"
    if path.exists() and not rebuild:
        bars = pd.read_parquet(path)
        bars.attrs.update({"timeframe": Timeframe.DAY.value, "adjusted": True})
        return bars
    if name in STOCK_DATASETS:
        bars = build_stock_dataset(name, provider, end)
    elif name in MACRO_DATASETS:
        bars = build_macro_dataset(name.split("_", 1)[1], provider, end)
    else:
        bars = build_rotation_dataset(name, provider, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    bars.to_parquet(tmp, index=False)
    tmp.replace(path)
    log.info("dataset %s: %d rows, %d columns", name, len(bars), bars["symbol"].nunique())
    return bars


def tradable_symbols(bars: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(s for s in bars["symbol"].astype(str).unique() if not s.startswith("^")))
