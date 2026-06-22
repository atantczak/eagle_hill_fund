"""Composition overlays: wrappers that take a strategy and return a modified one.

Volatility-targeting overlay, regime gate (Markov-switching on/off), and an HRP
allocator across multiple strategies. Built as wrappers so they compose onto any
Strategy. The composition seam is reserved here even though the overlays come
after the core engine.

Status: planned (seam reserved).
"""
