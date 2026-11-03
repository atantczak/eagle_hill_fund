# ADR-0011 — An idea, its study, its write-up and its runs are one thread

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

The first research question to arrive through the ideas box (Dad: leveraged ETFs
versus plain ones when the market turns bullish, and how long to hold the 3x fund)
exposed a gap. Write-ups lived in the library as PDFs with no link to the runs
their charts came from; runs lived in the lab with no link to the write-up that
explained them; ideas had a status but no way to reach the result. Andrew asked
for a precedent: when an idea is tested, there is a write-up, and the backtest
pages behind it say so and link back, "holistic and connected and seamless".

## Decision

One thread, readable from any end:

```
ideas/<id>.json ──link──► library/<slug>/meta.json ──runs[]──► var/results/<hash>
      ▲                      │ study                             ▲
      └──── ideas[] ─────────┘                                   │
                             └──────────── var/scorecards/<wf_id>.json (study, facts)
```

- **A tested question is a study.** `ehf/research/studies/<key>.py` plus a job
  in `jobs/` (and a row in the Studio table). Every scorecard the study writes
  carries `study: <key>` and a `facts` block for anything question-specific (the
  event table, the pair, the dataset fit). Its summary lands in
  `var/scorecards/<key>_<objective>_<stamp>.json`.
- **A write-up names what it is built on.** `meta.json` gains `study` (the key
  above), `runs` (registry hashes with a label: "the headline run", "hold QQQ")
  and `ideas` (ideas-box ids). `ops.py publish-report` takes `--study`, `--run
  HASH=label` and `--idea ID`; `ops.py link --slug` attaches them to a post that
  is already published, so the VIX write-ups get threaded too.
- **The site reads the thread backwards.** The post page shows "The backtests
  behind this write-up" (each run as a card with its headline numbers when the
  lab is mounted, otherwise just the link), the study's full scorecard table, and
  "Prompted by *idea* (*author*)". The run page shows the write-up card (named
  directly, or because the run belongs to the study the write-up reports on) and
  the idea behind it. The Studies table gets a study column with the write-up
  link. The ideas page shows "Read the result" once the idea has a `link`.
- **Report builders regenerate from data.** `scripts/build_<key>_report.py OUT
  IMG_DIR [STAMP]` reads only the study summary and the registry, in the
  explainer register (ADR-0006, and the tone note in `RESEARCH_PROGRAM.md`).
  Nothing about a write-up is hand-typed from a notebook.
- **The checklist is the precedent.** `docs/RESEARCH_PROGRAM.md` §7 "From an idea
  to a page" is the order of operations from here on.

## Consequences

- `Post` carries `runs`, `study`, `ideas`; `Idea` carries `link`. Missing fields
  mean an unthreaded post, which still renders exactly as before.
- The lab's run API makes one library listing per request (cached 60 s); a
  library hiccup degrades to "no write-up" rather than a broken page.
- An extended leveraged series (`TQQQ.X`) has no vendor, so the lab rebuilds it
  from its pair for the candle chart the same way the study did.
- The playground gained a third kind of strategy (a pair) next to dials and
  baskets.
