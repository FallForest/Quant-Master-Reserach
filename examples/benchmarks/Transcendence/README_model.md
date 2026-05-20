# RegimeHorizonCostEnsembleModel

## Hypothesis

The current CSI300 Alpha158 baseline is already strong for 1-step regression.
This candidate changes the target-and-selection layer instead of adding another wrapper:

1. train horizon-specialized learners (short/mid/long) using horizon-smoothed labels;
2. detect validation-time market regimes from prediction consensus/disagreement;
3. learn regime-specific blend weights with a cost-aware objective:
   mean(topk label) - turnover penalty - return-volatility penalty;
4. apply monotonic horizon weight control and score risk clipping.

## Implementation Path

Model file:

- `quant_master/contrib/model/regime_horizon_cost_ensemble.py`

Main class:

- `RegimeHorizonCostEnsembleModel`

Default base learners:

- `DEnsembleModel` on horizon 1
- `LGBModel` on horizon 5
- `LinearModel(ridge)` on horizon 10

## Leak-Safety Boundaries

1. horizon label construction is segment-local (`train` and `valid` built separately);
2. regime thresholds are fitted on validation predictions only;
3. blend weights and turnover memory boost are learned on validation only;
4. `test` phase only runs predict + frozen parameters.

## Cost-Aware Predict Path (Updated)

`RegimeHorizonCostEnsembleModel.predict` now applies `quant_master.contrib.strategy.topk_cost_aware.transform_scores_for_cost`
in a day-by-day loop, using only decision-time inputs:

- current day blended score;
- previous day selected holdings (for memory/turnover control);
- previous day score (for lagged volatility proxy construction).

This keeps predict-time `memory_boost`, `turnover_penalty`, and `risk_penalty` aligned
with the shared top-k cost-aware utility and avoids future information usage.

## Workflow Compatibility Layer

Current constructor is workflow-friendly and accepts compatibility fields used in
`Transcendence` YAMLs, including:

- `double_ensemble_kwargs`, `lightgbm_kwargs`, `linear_kwargs`
- `horizon_days`
- `turnover_penalty_grid`
- `risk_penalty_grid` and `cost_weight_grid` (mapped to risk penalty fallback)

These fields are mapped into real model behavior:

- base learner kwargs -> `horizon_model_specs`
- `horizon_days` -> per-horizon sub-model specs
- penalty grids -> validation-time penalty selection

Not-yet-implemented capabilities are accepted but explicitly reported via
`self.unused_config_keys` + logger warning, including:

- `secondary_handler`, `secondary_feature_set`, `feature_blend_mode`, `feature_weight_grid`
- `rolling_train_years`, `rolling_valid_months`
- `transformer_kwargs`

So current candidate does **not** consume secondary Alpha360 features yet; those
keys are reserved at config layer for next-stage implementation.

## Adapter Behavior Note

`_HorizonLabelDataset.prepare` now supports both:

- single segment (`"train"`, `"valid"`)
- multi-segment list/tuple (`["train", "valid"]`)

For multi-segment requests (used by `DEnsembleModel.fit`), horizon labels are now
replaced segment-by-segment, so DE/LGB/Linear branches share consistent horizon
target semantics.

## Minimal Config Notes

Use the class by full module path first (no registry edits required):

```yaml
model:
  class: RegimeHorizonCostEnsembleModel
  module_path: quant_master.contrib.model.regime_horizon_cost_ensemble
  kwargs:
    topk: 50
    search_step: 0.1
    turnover_penalty: 0.0002
    risk_penalty: 0.05
```

Tune candidate knobs in this order:

1. `horizon_model_specs` (horizons/model families)
2. `regime_*_quantiles`
3. `turnover_penalty`, `risk_penalty`, `memory_boost_grid`
4. `search_step` (0.1 -> 0.05 after first positive signal)

For a bounded validation search before full run, use:

- `workflow_config_regime_horizon_cost_ensemble_moderate_Alpha158_2026_csi300.yaml`
- quick/moderate smoke configs currently use two branches (`de_h1`, `lgb_h5`) for stability and runtime control.
