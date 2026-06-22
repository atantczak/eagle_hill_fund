"""Run specifications — the typed, hashable description of a backtest.

The ``config_hash`` is a deterministic SHA-256 over the canonical JSON of the
spec. It is the results-store primary key and the unit of reproducibility:
"give me this hash, regenerate the run." Code identity (git SHA) is recorded as
*metadata* on the run, NOT folded into the config hash — the same experiment run
from two commits is still the same experiment.

Costs and the fill model live in the spec and default to realistic (non-zero)
values, so a frictionless or look-ahead-prone run can only happen by explicit,
visible choice.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FillModel(StrEnum):
    """How/when orders fill relative to the signal bar."""

    NEXT_OPEN = "next_open"  # signal at close of t -> fill at open of t+1 (default, safe)
    NEXT_CLOSE = "next_close"
    SAME_CLOSE = "same_close"  # DEBUG ONLY: fills at the signal bar's close (look-ahead-ish)


class CostConfig(BaseModel):
    """Transaction-cost model. Defaults are realistic, not frictionless."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commission_per_share: float = 0.0  # ~$0 at Alpaca
    commission_min: float = 0.0
    slippage_bps: float = 1.0  # half-spread proxy, in basis points of price
    impact_coef: float = 0.0  # linear impact per unit of (order/ADV); 0 = off
    borrow_bps_annual: float = 0.0  # short borrow cost, annualized bps
    frictionless: bool = False  # explicit, labeled debug toggle ONLY

    def model_post_init(self, __context: Any) -> None:
        if self.frictionless and (self.slippage_bps or self.commission_per_share):
            # Frictionless must be unambiguous: zero everything else too.
            object.__setattr__(self, "slippage_bps", 0.0)
            object.__setattr__(self, "commission_per_share", 0.0)


class UniverseSpec(BaseModel):
    """Either an explicit symbol list or a named universe resolved at run time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: tuple[str, ...] = ()
    name: str | None = None  # e.g. "sp500"; resolved via DataProvider.get_universe

    def model_post_init(self, __context: Any) -> None:
        if not self.symbols and not self.name:
            raise ValueError("UniverseSpec needs either explicit symbols or a name")
        if self.symbols:
            object.__setattr__(self, "symbols", tuple(s.upper() for s in self.symbols))


class RunSpec(BaseModel):
    """Everything needed to reproduce one backtest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str  # registered strategy key, e.g. "rise_fall"
    params: dict[str, Any] = Field(default_factory=dict)
    universe: UniverseSpec
    start: str  # "YYYY-MM-DD"
    end: str
    timeframe: str = "1day"
    initial_capital: float = 1_000_000.0
    costs: CostConfig = Field(default_factory=CostConfig)
    fill_model: FillModel = FillModel.NEXT_OPEN
    sweep_id: str | None = None  # set when this run is a cell of a parameter sweep

    def canonical_json(self) -> str:
        """Deterministic JSON used for hashing (sorted keys, no whitespace drift)."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        """Stable SHA-256 of the spec. Primary key of the results store."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]
