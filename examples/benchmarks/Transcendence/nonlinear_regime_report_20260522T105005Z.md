# Nonlinear Regime Portfolio Search (20260522T105005Z)

## Notice
- Exploratory selection window: `2024-01-01..2026-04-30` (test-period selected parameters).
- Cost setting: `open=0.0005`, `close=0.0015`.

## Best Candidate
- Best(test exploratory): IR=3.028881, AnnRet=1.123950, MaxDD=-0.148526, Turnover=0.348410, hard_gate_pass=True
- Candidate ID: `5799429978545637032`
- Stage: `stage2`
- Conversion: `convex_softmax` / rebalance `weekly`

## 7406 / SOTA Comparison
- vs 7406: dIR=`0.228898`, dAnnRet=`0.879285`
- vs SOTA: dIR=`0.005879`, dAnnRet=`0.736096`

## Validation Replay
- Best-candidate pretest/year slices and walk-forward replay saved in artifacts.
