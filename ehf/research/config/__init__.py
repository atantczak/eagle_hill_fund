"""Typed configuration objects and run specs (pydantic).

A :class:`~ehf.research.config.spec.RunSpec` fully describes a backtest. Its
``config_hash`` is the primary key of the results store: identical spec ->
identical hash -> reproducible run.
"""

from ehf.research.config.spec import CostConfig, FillModel, RunSpec, UniverseSpec

__all__ = ["RunSpec", "CostConfig", "FillModel", "UniverseSpec"]
