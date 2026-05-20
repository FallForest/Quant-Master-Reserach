# Transcendence Validation-Selected Ensemble

## Scope

- Provider: `.qmData/cn_data`
- Train: `2012-01-01` to `2020-12-31`
- Valid (selection only): `2021-01-01` to `2023-12-31`
- Test (prediction/backtest only): `2024-01-01` to `2026-04-30`

## Why 7406 was strong

Compared with `773bd6d8413b4bb0b388a63a6b5b6a86`, `e2300230e0994a1a9ccbbd3bc4606d97`, `1a085ff9b5a34f408a44ad74055fc5da`:

- `7406` and `773` share the same DE core learner, but `7406` uses cost-aware objective settings (`topk=45`, `n_drop=4`, `turnover_penalty=5e-05`, `memory_boost_grid=[0,0.005]`) and wins on costed metrics.
- `e230` is pure DE baseline and loses mostly on execution-aware metrics.
- `1a085` hybrid has larger model complexity and underperforms in costed IR/AnnRet.

## Model implemented

- Class: `quant_master.contrib.model.transcendence_signal_ensemble.TranscendenceSignalEnsembleModel`
- Core behavior:
  - fit multiple base learners on train/valid
  - compute base predictions on valid
  - select ensemble weights and execution transform params only on valid
  - test stage only calls `predict(test)` and backtest never feeds test labels to selector

## Selection logic (valid-only)

1. Build candidate blend weights (equal, one-hot, simplex/random candidates).
2. Quick valid screening to keep top candidate weights.
3. Full valid objective search over:
   - weight candidate
   - `topk_grid`, `n_drop_grid`
   - `memory_boost_grid`, `turnover_penalty_grid`, `volatility_penalty_grid`
4. Select best objective state and freeze for test prediction.

## Runs

- Main config:
  - `examples/benchmarks/Transcendence/workflow_config_transcendence_validation_selected_ensemble_Alpha158_2026_csi300.yaml`
  - run_id: `0d7d238af8dc4dadae6daf38993b0302`
- Fallback config:
  - `examples/benchmarks/Transcendence/workflow_config_transcendence_validation_selected_ensemble_fallback_Alpha158_2026_csi300.yaml`
  - run_id: `352791cc842043da92643a0fb276df53`

Detailed evidence is in:

- `examples/benchmarks/Transcendence/ensemble_run_validation_selected_20260520T153806Z.json`
- `examples/benchmarks/Transcendence/ensemble_run_validation_selected_20260520T153806Z.csv`
- `examples/benchmarks/Transcendence/ensemble_run_validation_selected_20260520T153806Z.md`
