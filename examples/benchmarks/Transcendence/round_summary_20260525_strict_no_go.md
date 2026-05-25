# Round Summary: Strict NO-GO (Q-ROUND-SUMMARY-20260525)

Date: 2026-05-25

## Strict gate

- Evaluation window: 2024-01-01..2026-04-30.
- Costs: open_cost=0.0005, close_cost=0.0015.
- Selection rule: parameters must be selected outside the test window.
- Report integrity: complete finite daily report required.
- Pass thresholds: IR > 2.90 and AnnRet > 0.27.

## Old candidate reruns

Both old candidates failed the strict gate.

| Candidate | Status | IR | AnnRet | Finite daily rows | Nonfinite |
| --- | --- | ---: | ---: | ---: | ---: |
| alpha360_de_only | NO-GO | 1.3438666890 | 0.1755958479 | 562/562 | 0 |
| alpha158_regime_de_only | NO-GO | 1.1968474211 | 0.1493904865 | 562/562 | 0 |

## New label prototypes

All new label prototypes showed validation attraction but failed to transfer to the full 2024-2026 strict window. All full reports are 562/562 finite with nonfinite=0.

| Prototype | Valid IR | Valid AnnRet | Full IR | Full AnnRet | Full finite daily rows | Nonfinite | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| drawdown conditional | 2.482617 | 0.191759 | 0.113301 | 0.012299 | 562/562 | 0 | NO-GO |
| liquidity survival | 2.693506 | 0.234283 | 0.111855 | 0.012361 | 562/562 | 0 | NO-GO |
| market-state target | 2.821698 | 0.192039 | 0.098760 | 0.009667 | 562/562 | 0 | NO-GO |

## Protocol fix

- liquidity_survival full_hard_gate previously checked only finite rows.
- The gate was fixed to also require IR > 2.90 and AnnRet > 0.27.
- After rerun, liquidity_survival gatefix full result is NO-GO with full IR=0.111855 and AnnRet=0.012361.
- The earlier 20260525T072112Z liquidity summary is superseded by the gatefix rerun and must not be treated as a pass artifact.

## Near-line base40/gru45 review

| Line | AnnRet | IR | Decision |
| --- | ---: | ---: | --- |
| base40 | 0.2704249361 | 2.8063802748 | Stop |
| gru45 | 0.2446295612 | 2.9447364812 | Stop |

The base40/gru45 line has no legal pre-2024 overlap selection, so it is stopped despite being near one of the two strict thresholds.

## Conclusion

No robust leap-ahead model crossed the strict gate in this round. The 2023 validation attraction did not migrate to the 2024-2026 strict evaluation window. The next credible path is to introduce real new data sources, or rebuild labels/targets with a stronger requirement for multi-year pre-2024 stability before any 2024-2026 test-window claim.

## Key artifacts

- examples/benchmarks/Transcendence/model/double_ensemble_alpha360_runner.py
- artifacts/hard_gate_pass/double_ensemble_alpha360_alpha360_de_only_rerun_alpha360_de_only_full_summary_20260525T063118Z.json
- artifacts/hard_gate_pass/double_ensemble_alpha360_alpha158_regime_de_only_rerun_alpha158_regime_de_only_full_summary_20260525T064811Z.json
- examples/benchmarks/Transcendence/support/drawdown_conditional_label_pre2024_summary_20260525T070734Z.md
- examples/benchmarks/Transcendence/support/drawdown_conditional_label_pre2024_summary_20260525T070734Z.json
- examples/benchmarks/Transcendence/support/liquidity_survival_label_pre2024_gatefix_summary_20260525T073203Z.md
- examples/benchmarks/Transcendence/support/liquidity_survival_label_pre2024_gatefix_summary_20260525T073203Z.json
- examples/benchmarks/Transcendence/support/market_state_target_pre2024_summary_20260525T073021Z.md
- examples/benchmarks/Transcendence/support/market_state_target_pre2024_summary_20260525T073021Z.json
- examples/benchmarks/Transcendence/archive/runs/replay_action_reports_cache_20260524T045740Z/replay_action_reports_cache_summary_20260524T045740Z.json
- examples/benchmarks/Transcendence/archive/runs/regime_switch_strategy_replay_probe_base40_gru45_20260524T052522Z/regime_switch_strategy_replay_probe_base40_gru45_summary_20260524T052522Z.json
