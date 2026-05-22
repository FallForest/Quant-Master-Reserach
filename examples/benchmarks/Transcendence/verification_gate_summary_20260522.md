# Transcendence Verification Gate Summary (Worker D)

- Verifier time: 2026-05-22 (Asia/Shanghai)
- Scope: read-only verification of existing artifacts under `examples/benchmarks/Transcendence`
- Hard gate (net-cost, test period): `2024-01-01..2026-04-30`, `IR > 2.90` and `AnnRet > 0.27`

## Baseline / SOTA Snapshot

| Item | Source artifact | IR | AnnRet | MDD | Turnover | Notes |
|---|---|---:|---:|---:|---:|---|
| Baseline 7406 default tk45/nd4 daily | `portfolio_fixed_oos_summary_7406e470_20260520T070317Z.json` (`baseline_full_period`) | 2.7999836767 | 0.2446646361 | -0.0481259850 | 0.1749957877 | Reference baseline |
| Strategy-scan SOTA buffered tk55/hk85 weekly | `sota_snapshot.json` + `portfolio_fixed_oos_summary_7406e470_20260520T070317Z.json` | 3.0230019402 | 0.3878544155 | -0.0477231243 | 0.2547139624 | Parameters selected on test period (high overfit risk) |
| Regime overlay (continuous selector) | `regime_switch_stability_reverify_summary_7406e470_20260520T105141Z.json` | 2.9565999209 | 0.2762009151 | -0.0481259850 | 0.1749957877* | *turnover field not emitted for regime full; baseline turnover used as reference only |

## Candidate Gate Results

| Candidate | Quant hard gate | Engineering gate | Test gate | Leakage gate | Performance gate | Result |
|---|---|---|---|---|---|---|
| broad tinyA (`rank_ensemble_scan_broad_tinyA_summary_20260520T112550Z`) | **FAIL** (`IR=2.8965721768 < 2.90`) | PASS (artifact complete) | PASS (summary+candidate+csv) | WARNING (explicit test-period selection notice) | PASS | **NO-GO** |
| regime overlay (`regime_switch_stability_reverify_summary_7406e470_20260520T105141Z`) | **PASS** (`IR=2.9565999209`, `AnnRet=0.2762009151`) | PASS | PASS (reverify artifact consistent with initial run) | PASS (t-1, expanding quantile, fixed-rule declarations present) | PASS | **GO (conditional)** |
| strategy-scan buffered SOTA (`fixed_buffered_tk55_hk85_equal_weekly`) | PASS by headline metrics | PASS | PASS | **FAIL/WARNING** (test-period parameter search acknowledged in snapshot) | PASS | **NO-GO for integration as “new breakthrough”** |

## Independent Recheck Notes

1. Regime overlay initial and reverify summaries are numerically consistent (differences only at floating precision scale).
2. broad tinyA does not meet the hard threshold despite beating old 7406 IR.
3. Buffered strategy remains highest reported IR/AnnRet but is explicitly marked as test-period-selected in `sota_snapshot.json`; cannot be treated as robust integrable breakthrough without walk-forward or locked OOS validation.

## Final Go/No-Go

- Integrable breakthrough now: **Only regime overlay is conditionally GO** (meets hard threshold and has leakage guardrail declarations + reverify consistency).
- Portfolio/strategy SOTA buffered variant: **NO-GO** until out-of-sample/forward validation removes parameter-selection leakage risk.
- broad tinyA and other scanned blends: **NO-GO** under current hard gate.

## Command Evidence (read-only)

- `rg -n "7406|tinyA|regime|SOTA|baseline|IR|AnnRet|2024-01-01|2026-04-30" examples/benchmarks/Transcendence`
- `Get-ChildItem -Path examples/benchmarks/Transcendence -Recurse -File | Where-Object { $_.Extension -in '.json','.md','.csv' }`
- `Get-Content examples/benchmarks/Transcendence/sota_snapshot.json`
- `Get-Content examples/benchmarks/Transcendence/portfolio_fixed_oos_summary_7406e470_20260520T070317Z.json`
- `Get-Content examples/benchmarks/Transcendence/rank_ensemble_scan_broad_tinyA_summary_20260520T112550Z.json`
- `Get-Content examples/benchmarks/Transcendence/rank_ensemble_scan_broad_tinyA_candidate_20260520T112550Z.json`
- `Get-Content examples/benchmarks/Transcendence/regime_switch_stability_summary_7406e470_20260520T094309Z.json`
- `Get-Content examples/benchmarks/Transcendence/regime_switch_stability_reverify_summary_7406e470_20260520T105141Z.json`
