"""EHF Complexity Core — Santa Fe-tradition complexity theory over markets.

A module that *consumes* the EHF platform (data, backtest, analysis, results),
never a parallel project. Layers, bottom-up:

  measures/     PURE measures (entropy, memory, information) — the DRY core
  reference     known systems (white noise, AR(1), Lorenz, Hénon, coupled) for
                ground-truth validation of every measure
  networks/     dependency graphs & rolling topology         (planned)
  agents/       agent-based simulation substrate + worlds     (planned)
  evolution/    one adaptive-search engine (GA/GP)            (planned)
  landscapes/   NK fitness landscapes + ruggedness diagnostics (planned)
  synthetic/    ABM-generated price paths + stylized-facts validator + a
                platform-wide robustness harness              (planned)
  applications/ THIN declarations emitting standard RunResults (planned)

Distrust elegance: every application faces the same validation gauntlet as any
strategy (cost-aware backtest, deflated Sharpe, OOS walk-forward). A complexity
model that can't beat the cost-aware benchmark is a physics demo, not alpha.
"""
