"""Eagle Hill Fund — one package, three things.

Layout::

    ehf/
      research/      the quant research platform (data, backtests, strategies, reports)
      web/           eaglehillfund.com (FastAPI over the reports bucket)
      integrations/  Financial Modeling Prep client the research data layer wraps
      tools/         legacy shared utilities the FMP client depends on
      tests/

Research is the product; ``web`` publishes what it produces. ``integrations`` and
``tools`` are kept only because ``research.data.fmp_provider`` still imports them.
See ``docs/ARCHITECTURE.md``.
"""

__version__ = "0.1.0"
