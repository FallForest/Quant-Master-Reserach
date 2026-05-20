# Ensemble Run Evidence (Validation-Selected)

- timestamp_utc: `2026-05-20T15:38:06Z`
- target_sota_run: `7406e47063e9479cb34d300b9ed03bad`
- target_sota: `IR=2.799983676714277`, `AnnRet=0.24466463608994535`

## 7406 vs 773/e230/1a085 (read-only audit)

| run_id | model | costed_ir | costed_annret | ic | rank_ic |
|---|---|---:|---:|---:|---:|
| `7406e47063e9479cb34d300b9ed03bad` | RegimeHorizonCostEnsemble (DE-only + cost-aware exec) | 2.799984 | 0.244665 | 0.023921 | 0.021225 |
| `773bd6d8413b4bb0b388a63a6b5b6a86` | RegimeHorizonCostEnsemble (DE-only baseline exec) | 2.515274 | 0.211376 | 0.025011 | 0.022773 |
| `e2300230e0994a1a9ccbbd3bc4606d97` | DEnsemble baseline | 2.351364 | 0.204507 | 0.023859 | 0.021645 |
| `1a085ff9b5a34f408a44ad74055fc5da` | TranscendenceHybrid | 1.494812 | 0.127166 | 0.020344 | 0.023018 |

结论：`7406` 主要强在 execution-aware valid 选参与成本约束（topk=45/n_drop=4/turnover penalty + memory boost），而非单纯 DE 训练误差优势。

## Implemented validation-selected ensemble

- model: `quant_master.contrib.model.transcendence_signal_ensemble.TranscendenceSignalEnsembleModel`
- workflow(main): `workflow_config_transcendence_validation_selected_ensemble_Alpha158_2026_csi300.yaml`
- workflow(fallback): `workflow_config_transcendence_validation_selected_ensemble_fallback_Alpha158_2026_csi300.yaml`

valid-only 选权逻辑：

1. fit 期训练全部 base learners（train/valid）。
2. 仅在 valid 段构造候选权重（equal/onehot/simplex/random）并做 quick filter。
3. 仅在 valid 段搜索 `weights + topk/n_drop + memory_boost/turnover_penalty/volatility_penalty` 目标。
4. test 段只预测回测，不读取 test label。

## Full runs (test: 2024-01-01 ~ 2026-04-30)

| tag | run_id | costed_ir | costed_annret | ic | rank_ic | maxdd | turnover | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_main | `0d7d238af8dc4dadae6daf38993b0302` | 1.792948 | 0.154577 | 0.021006 | 0.021417 | -0.059101 | 0.176858 | 1174.342 |
| ensemble_fallback | `352791cc842043da92643a0fb276df53` | 2.274719 | 0.193321 | 0.024145 | 0.022584 | -0.052417 | 0.175621 | 1035.127 |

selected weights / params:

- `0d7d...`
  - weights: `regime_de_7406=0.152743, de_baseline=0.291283, lgb_aux=0.371914, lin_aux=0.184061`
  - selected: `topk=45, n_drop=4, memory_boost=0.0, turnover_penalty=0.0, volatility_penalty=0.0`
- `3527...`
  - weights: `regime_7406=0.333333, regime_773=0.333333, de_baseline=0.333333`
  - selected: `topk=45, n_drop=4, memory_boost=0.0, turnover_penalty=0.0, volatility_penalty=0.0`

## SOTA decision

- best_new_run: `352791cc842043da92643a0fb276df53`
- gate: `IR > 2.799983676714277` and `AnnRet >= 0.24466463608994535`
- outcome: **not passed**
- `candidate_model_sota = false`
