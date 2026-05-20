# Round Summary: R7406

## Identity

- round_id: `R7406`
- run_id: `7406e47063e9479cb34d300b9ed03bad`
- model_name: `RegimeHorizonCostEnsembleModel`
- workflow_config: `examples/benchmarks/Transcendence/workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml`
- owner: `Model/Experiment Worker`
- timestamp_utc: `2026-05-19T11:11:06Z`

## Command

```powershell
py -3 -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml
```

## Core Metrics

| metric | value |
| --- | ---: |
| IC | `0.023921221367623043` |
| RankIC | `0.021225039288402008` |
| costed AnnRet | `0.24466463608994535` |
| costed IR | `2.799983676714277` |
| max drawdown | `-0.04812598495553819` |
| turnover | `0.17499578770011953` |
| runtime_sec | `412.693` |
| leakage_check | `pass` |

## Relative Gate (vs README baseline)

Reference baseline: IC `0.024251`, RankIC `0.023848`, AnnRet `0.147324`, IR `1.935775`.

| item | delta | pass |
| --- | ---: | --- |
| delta_ic | `-0.00032977863237695915` | `no` |
| delta_rank_ic | `-0.002622960711597993` | `no` |
| delta_costed_annret | `+0.09734063608994534` | `yes` |
| delta_costed_ir | `+0.8642086767142771` | `yes` |

## Absolute Gate

| item | threshold | value | pass |
| --- | ---: | ---: | --- |
| IC | `>= 0.020000` | `0.023921221367623043` | `yes` |
| RankIC | `>= 0.020000` | `0.021225039288402008` | `yes` |
| costed AnnRet | `>= 0.120000` | `0.24466463608994535` | `yes` |
| costed IR | `>= 1.600000` | `2.799983676714277` | `yes` |
| max drawdown | `<= 0.250000` | `0.04812598495553819` | `yes` |
| turnover | `<= 0.600000` | `0.17499578770011953` | `yes` |
| runtime_sec | `<= 43200` | `412.693` | `yes` |
| leakage_check | `== pass` | `pass` | `yes` |

## Decision

- status: `portfolio_sota_candidate`
- final_decision: `GO (portfolio gate) / PARTIAL (strict all-signal gate)`
- rationale:
  - Portfolio metrics are materially stronger than README baseline (costed IR and AnnRet both significantly higher).
  - Signal-relative gate is partial because IC and RankIC are below README baseline.
  - Candidate is accepted as portfolio-SOTA candidate only, not strict all-signal SOTA.

## Residual Risks

- Signal robustness risk: RankIC and IC relative regressions could hurt confidence in cross-sectional ranking stability.
- Cost/turnover regime sensitivity: gains are portfolio-centric and may be sensitive to transaction-cost assumptions.
- Governance risk: this run should not be labeled strict SOTA without recovering IC and RankIC relative deltas.

## Artifact Checklist

- [x] leaderboard row added
- [x] round summary saved
- [x] sota snapshot updated
- [x] command evidence attached

## Addendum: Regime Switch Stability Verification (2026-05-20)

- candidate_artifact: `examples/benchmarks/Transcendence/regime_switch_stability_summary_7406e470_20260520T094309Z.json`
- scope: strategy/regime-switch over existing signals (no new model retraining)
- full-period verified metrics (`2024-01-02` to `2026-04-30`):
  - regime switch: costed IR `2.956599920926148`, costed AnnRet `0.2762009150931481`
  - base 7406: costed IR `2.799983676714282`, costed AnnRet `0.2446646360899453`
- leakage review: pass (decision diagnostics use t-1 windows; expanding thresholds use history through t-1; single continuous rule over full horizon; no per-slice future selection)
- rerun evidence:
  - command: `py -3 examples/benchmarks/Transcendence/regime_switch_stability_eval.py --run-id 7406e47063e9479cb34d300b9ed03bad --tracking-uri file:./mlruns --open-cost 0.0005 --close-cost 0.0015 --output-prefix regime_switch_stability_reverify`
  - reverified full metrics: costed IR `2.9565999209261435`, costed AnnRet `0.27620091509314726` (floating-point equivalent)
- promotion decision: accepted as verified strategy candidate, not promoted to current portfolio SOTA (current SOTA IR/AnnRet `3.0230019401859436` / `0.3878544154715252` remains higher).
