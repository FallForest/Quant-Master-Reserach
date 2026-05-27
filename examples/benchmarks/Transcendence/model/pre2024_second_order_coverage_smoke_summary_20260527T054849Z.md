# Pre-2024 Second-Order Coverage Smoke

- timestamp_utc: `2026-05-27T05:48:49Z`
- verdict: `NO_GO`
- gate_pass: `False`
- legal_pre2024_selection_possible: `False`
- no_2024_plus_used_for_gate: `True`
- eligible_complete_signals: `0`
- max_abs_pairwise_corr: `None`

## Coverage

| key | status | min_date | max_date | rows_pre2024_finite | days | names | 2020 | 2021 | 2022 | 2023 | complete | first_2024+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| factor_augmented_meta | ok | 2024-01-02 | 2024-01-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2024-01-02 |
| factor_meta_gru_base_fusion_lockstep_patchrun | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_gru_base_fusion_lockstep | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_gru_base_fusion_lockstep | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_gru_base_fusion_lockstep | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_gru_base_fusion_lockstep_verify | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_stability_refit | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_stability_refit | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| factor_meta_stability_refit | skipped_pkl_no_csv |  |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| long_history_retrain | ok | 2024-01-02 | 2024-01-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2024-01-02 |
| long_history_second_order | ok | 2025-01-02 | 2025-01-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2025-01-02 |

## Pairwise Correlation

No compliant complete 2020-2023 signal pair was available for correlation.

## Recommendation

Do not continue to weight selection: no two distinct compliant signals with complete 2020-2023 coverage and pairwise rank correlation below 0.85 were found. Generate a low-cost csi300 2020-2021 OOS panel only if the lead approves a bounded prediction job.
