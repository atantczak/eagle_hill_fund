# EHF Complexity Core — Module Guide

A complexity-theory module that **consumes** the EHF platform (data, signals,
backtest, results) — never a parallel project. Santa Fe lineage: Arthur's
El Farol / Minority Game, Holland's adaptive agents & GAs, Kauffman's NK
landscapes, the Farmer/Packard "Prediction Company" tradition.

> **Status legend:** ✅ built & tested · 🟡 next · ⚪ planned

## Layers (bottom-up)

```
ehf/server/research/complexity/
  reference.py   ✅ known systems (white noise, AR(1), random walk, Lorenz, Hénon,
                    coupled X->Y) — ground truth every measure must recover
  measures/      ✅ PURE measures (the DRY core, analog of ehf/server/research/signals)
                    entropy.py     : permutation_entropy, shannon_entropy
                    memory.py      : hurst (R/S), dfa
                    information.py : transfer_entropy, effective_transfer_entropy,
                                     mutual_information, transfer_entropy_matrix
  networks/      ⚪ correlation/TE dependency graphs, Mantegna MST, rolling topology
  agents/        ⚪ Agent ABC + Market + sim loop; Minority Game, Santa Fe ASM
  evolution/     ⚪ one GA/GP engine (evolve a rule | calibrate an ABM)
  landscapes/    ⚪ NK landscapes + ruggedness -> overfitting-risk score
  synthetic/     ⚪ ABM -> price paths + stylized-facts validator + robustness harness
  applications/  ⚪ THIN declarations emitting standard RunResults via the registry
```

## Reuse contract (harmony with the platform)

- Measures are **pure** (numpy only) and validated against `reference.py` — the
  same discipline as `ehf/server/research/signals`.
- Regime measures are **wired into `ehf.server.research.signals`**: `signals.rolling_permutation_entropy`
  and `signals.rolling_dfa` (rolling Hurst already exists as `signals.hurst`), so
  a strategy consumes them like any other signal.
- Applications (when built) will emit the **same `RunResult`** through the **same
  `ExperimentRegistry`** and tearsheets as any strategy. No bespoke result formats.
- **Distrust elegance:** every application faces the same gauntlet as any strategy
  (cost-aware backtest, deflated Sharpe, OOS walk-forward). A complexity model that
  can't beat the cost-aware benchmark is a physics demo, not alpha — labelled as such.

## Acceptance gauntlet (reproducible — `tests/test_complexity_measures.py`)

Every measure recovers ground truth before it touches price data:

| Measure | white noise | AR(1) +0.9 | AR(1) −0.6 | random walk | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| Permutation entropy (o=4) | **1.00** | 0.95 | 0.95 | — | **0.29** | 0.69 |
| Hurst (R/S) | 0.57 | **0.77** | **0.46** | — | 0.87 | — |
| DFA exponent α | 0.52 | 0.81 | — | **1.48** | — | — |

Transfer entropy on the coupled reference X→Y (c=0.7): raw TE(X→Y)=0.41 ≫
TE(Y→X)=0.02; **effective** (bias-corrected) eTE(X→Y)=0.39, eTE(Y→X)≈0.00, and
independent series → eTE≈0.00. Mutual information: I(x;x)=3.0=log₂8, I(indep)≈0.

Interpretation: bold cells are the discriminating signals — maximal ordinal
entropy for noise vs. **0.29 for deterministic chaos**; R/S separates
persistent/anti-persistent; DFA recovers the textbook **1.5** for integrated
noise; effective TE recovers *direction* and kills finite-sample bias.

## First live result — daily TE lead-lag (honest finding)

Rolling-effective TE matrix over {SPY, AAPL, MSFT, NVDA, JPM, XLF}, daily
2022–2024 (≈500 obs), pulled live through `FMPProvider`:

- The estimator works end-to-end on real data (FMP → canonical → cache → Panel →
  returns → TE matrix).
- **Detected daily flows are weak and near the noise floor** (≤~0.04 bits).
  Financials (JPM, XLF) show the largest net *outflow*; SPY's net flow ≈ 0.
- **The expected "index → components" lead does NOT appear at daily frequency** —
  and that's the honest truth, not an estimator failure: daily index/component
  moves are near-*contemporaneous*, so directed *lagged* information is small. The
  synthetic test proves the estimator finds strong directed flow when it genuinely
  exists. Lead-lag is an **intraday** phenomenon → a real study should use
  `Timeframe.HOUR`/`MINUTE` bars (the provider supports them) and longer history.

This is exactly the "distrust elegance" posture: report what the data says, not
what the theory hopes.

## Build order (remaining)

1. **Application #1** — permutation-entropy / Hurst **regime gate** on a trend
   strategy, through the existing engine (first end-to-end complexity result). 🟡
2. `networks/` + **stress gate** (rolling topology → risk-off). ⚪
3. `agents/` + **Minority Game** (validate the α=2^m/N phase transition) + MG
   directional predictor. ⚪
4. **Santa Fe ASM** → `synthetic/` + stylized-facts validator + a platform-wide
   **robustness harness** (re-run any strategy across N synthetic histories). ⚪
5. `evolution/` + `landscapes/` + **evolved-rule strategy** with a ruggedness-based
   overfitting-risk score. ⚪
