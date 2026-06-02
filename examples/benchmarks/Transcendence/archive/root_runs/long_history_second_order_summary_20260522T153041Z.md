# Long History Second Order Ensemble (20260522T153041Z)

## Verdict
- `NO_GO`

## Protocol
- Main leg: `factor_augmented_meta_candidate_pred_20260522T120515Z.pkl`.
- Auxiliary leg: `long_history_retrain_candidate_pred_20260522T134241Z.pkl`.
- Residualization: daily cross-sectional linear residual of long_history versus main leg, then rank-center.
- Selection: quarterly forward, weight grid only, prior-only blocked folds.
- Effective evaluation window: `2025-01-01..2026-04-28`.

## Same-Window Backtest
- Candidate backtest ok: `False`
- Candidate backtest metrics: `FAILED`
- Candidate backtest error: `{"type": "ValueError", "message": "only have 107734.1514283342 SZ300661, require 107734.34572806784"}`
- Factor meta backtest ok: `True`
- Factor meta backtest metrics: `{"annret": 0.14100083163363758, "ir": 1.55735205227312, "max_drawdown": -0.037867919347507786, "turnover": 0.23194079030559514, "elapsed_sec": 2.130612600000859}`
- Factor meta backtest error: ``
- Candidate proxy: ICIR=`2.378954` AnnProxy=`10.908120` MDDProxy=`-0.911684` TurnoverProxy=`0.507198`
- Factor meta proxy: ICIR=`2.410728` AnnProxy=`10.933324` MDDProxy=`-0.909240` TurnoverProxy=`0.500086`
- Proxy delta vs factor meta: ICIR=`-0.031774` AnnProxy=`-0.025205`

## Selected Weights By Quarter

| quarter | apply_start | apply_end | selected_weight | cv_folds | cv_score |
|---|---|---|---:|---:|---:|
| 2024Q1 | 2024-01-01 | 2024-03-31 | 0.00 | 0 |  |
| 2024Q2 | 2024-04-01 | 2024-06-30 | 0.00 | 0 |  |
| 2024Q3 | 2024-07-01 | 2024-09-30 | 0.00 | 0 |  |
| 2024Q4 | 2024-10-01 | 2024-12-31 | 0.15 | 1 | 8.375112 |
| 2025Q1 | 2025-01-01 | 2025-03-31 | 0.15 | 2 | 2.779250 |
| 2025Q2 | 2025-04-01 | 2025-06-30 | 0.15 | 3 | 3.839723 |
| 2025Q3 | 2025-07-01 | 2025-09-30 | 0.20 | 3 | 2.173792 |
| 2025Q4 | 2025-10-01 | 2025-12-31 | 0.00 | 3 | 2.745037 |
| 2026Q1 | 2026-01-01 | 2026-03-31 | 0.00 | 3 | 5.644119 |
| 2026Q2 | 2026-04-01 | 2026-04-28 | 0.20 | 3 | 6.997649 |

## Notes
- Common signal coverage ends on `2026-04-28`, so this smoke run stops on `2026-04-28`.
- Published factor_meta full-window reference remains IR=`2.820878` / AnnRet=`0.338626` on `2024-01-01..2026-04-30`.