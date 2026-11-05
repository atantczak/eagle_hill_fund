"""The ``Strategy`` ABC and registry.

A strategy declares intent in three overridable steps; the engine consumes only
``target_weights``:

    generate_signals(panel)  -> raw signal/score per (bar, symbol)
    size_positions(signals)  -> target weights      (default: pluggable sizing)
    manage_exits(weights)    -> target weights       (default: identity)

Subclasses are thin (~30-80 lines): pick an entry signal from ``ehf.research.signals``,
a sizing component from ``ehf.research.strategies.sizing``, and exit rules from
``ehf.research.strategies.exits``. Shared logic is never copy-pasted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ehf.research.backtest.panel import Panel
from ehf.research.strategies import sizing

_REGISTRY: dict[str, type[Strategy]] = {}


def register(key: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a strategy under ``key`` (its RunSpec name)."""

    def deco(cls: type[Strategy]) -> type[Strategy]:
        if key in _REGISTRY:
            raise ValueError(f"strategy key {key!r} already registered")
        cls.key = key
        _REGISTRY[key] = cls
        return cls

    return deco


def build_strategy(key: str, params: dict[str, Any]) -> Strategy:
    """Instantiate a registered strategy from its key + params."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**params)


class Strategy(ABC):
    """Base class. Override the three hooks; keep it thin."""

    key: str = ""
    #: default parameters; subclasses override. Used to fill missing params.
    defaults: dict[str, Any] = {}

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = {**self.defaults, **params}

    def p(self, name: str) -> Any:
        """Look up a (defaulted) parameter."""
        return self.params[name]

    @abstractmethod
    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        """Raw signal/score per (bar, symbol), using only data through each bar."""

    def size_positions(self, signals: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        """Convert signals to target weights. Default: gross-normalized."""
        return sizing.normalize_weights(signals, gross=self.params.get("gross_exposure", 1.0))

    def manage_exits(self, weights: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        """Apply exit rules. Default: identity (exits encoded upstream)."""
        return weights

    def target_weights(self, panel: Panel) -> pd.DataFrame:
        """Template method the engine calls: signals -> size -> exits."""
        signals = self.generate_signals(panel)
        weights = self.size_positions(signals, panel)
        weights = self.manage_exits(weights, panel)
        every = self.params.get("rebalance_every", 1)
        band = self.params.get("rebalance_band", 0.0)
        if (every and every > 1) or (band and band > 0):
            from ehf.research.overlays.rebalance import throttle

            weights = throttle(weights, every=every, band=band)
        return weights
