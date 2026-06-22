"""Eagle Hill Fund — unified codebase.

Layout::

    ehf/
      client/                front-end (placeholder for now)
      server/
        research/            the quant research & trading platform
        integrations/        external services (Financial Modeling Prep, AWS, ...)
        tools/               shared utilities (parallelization, HTTP client, portfolio)

The research platform is the product; ``integrations`` and ``tools`` are the
infrastructure it builds on. See ``docs/ARCHITECTURE.md``.
"""

__version__ = "0.1.0"
