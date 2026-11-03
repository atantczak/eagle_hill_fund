"""Daily market research report: screens -> stored snapshot -> narrative -> email.

The report is a *screen* over the whole liquid US universe at one point in time
(not a backtest). It is built in layers so each is testable on its own:

  config       ReportConfig: thresholds, universe filters, delivery targets (env-driven)
  fmp          thin FMP client for the report (bulk CSVs, batch quotes, bars, feeds)
  universe     which symbols to screen (liquid screener ∪ S&P 500)
  features     pure price features per symbol from one year of daily bars
  fundamentals pure column mapping of the FMP bulk fundamentals
  screens      pure rules -> ranked ``Hit`` records (the "signals" of the report)
  market       pure regime / breadth / macro summary
  snapshot     orchestration: fetch everything -> ``Snapshot``
  narrative    optional Claude write-up over the structured snapshot
  render       markdown + HTML
  deliver      S3 publish + SES email
  job          ``run()`` + CLI;  lambda_handler  the AWS entry point
  evaluate     forward-return scorecard of past hits (the point of storing them)

Only ``fmp``, ``narrative``, ``deliver``, ``snapshot``, ``job`` and ``evaluate``
touch the network. Everything else is pure and unit-tested.

Status: v1 (2026-11).
"""
