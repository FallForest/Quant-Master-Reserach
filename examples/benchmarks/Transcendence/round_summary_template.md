# Round Summary: <round_id>

## Identity

- round_id: `<round_id>`
- run_id: `<run_id>`
- model_name: `<model_name>`
- workflow_config: `<workflow_config_path>`
- owner: `Worker D / <name>`
- timestamp_utc: `<YYYY-MM-DDTHH:MM:SSZ>`

## Command

```powershell
<exact command>
```

## Core Metrics

| metric | value |
| --- | ---: |
| IC | `<ic>` |
| RankIC | `<rank_ic>` |
| costed AnnRet | `<costed_annret>` |
| costed IR | `<costed_ir>` |
| max drawdown | `<max_drawdown>` |
| turnover | `<turnover>` |
| runtime_sec | `<runtime_sec>` |
| leakage_check | `<pass/fail/unknown>` |

## Relative Gate (vs baseline or current SOTA)

| item | delta | pass |
| --- | ---: | --- |
| delta_ic | `<ic - ref_ic>` | `<yes/no>` |
| delta_rank_ic | `<rank_ic - ref_rank_ic>` | `<yes/no>` |
| delta_costed_annret | `<costed_annret - ref_costed_annret>` | `<yes/no>` |
| delta_costed_ir | `<costed_ir - ref_costed_ir>` | `<yes/no>` |

## Absolute Gate

| item | threshold | value | pass |
| --- | ---: | ---: | --- |
| IC | `>= 0.020000` | `<ic>` | `<yes/no>` |
| RankIC | `>= 0.020000` | `<rank_ic>` | `<yes/no>` |
| costed AnnRet | `>= 0.120000` | `<costed_annret>` | `<yes/no>` |
| costed IR | `>= 1.600000` | `<costed_ir>` | `<yes/no>` |
| max drawdown | `<= 0.250000` | `<max_drawdown>` | `<yes/no>` |
| turnover | `<= 0.600000` | `<turnover>` | `<yes/no>` |
| runtime_sec | `<= 43200` | `<runtime_sec>` | `<yes/no>` |
| leakage_check | `== pass` | `<leakage_check>` | `<yes/no>` |

## Decision

- status: `<smoke_passed/full_passed/accepted_sota/rejected/invalid>`
- final_decision: `<GO/NO-GO>`
- rationale: `<2-5 lines>`

## Artifact Checklist

- [ ] leaderboard row added
- [ ] round summary saved
- [ ] sota snapshot updated (only if accepted_sota)
- [ ] command evidence attached
