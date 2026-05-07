# BeatDoubleEnsemble

This benchmark track targets the current local DoubleEnsemble baseline:

- Universe: CSI300
- Handler: Alpha158
- Train: 2012-01-01 to 2020-12-31
- Valid: 2021-01-01 to 2023-12-31
- Test/backtest: 2024-01-01 to 2026-04-30
- Strategy: TopkDropoutStrategy, topk=50, n_drop=5
- Primary metric: excess return with cost, information ratio

The current completed baseline report is:

- IC: 0.024251
- Rank IC: 0.023848
- Excess annualized return with cost: 0.147324
- Excess information ratio with cost: 1.935775

## Experiment Matrix

| Priority | Experiment | Hypothesis | Implementation | Success check |
| --- | --- | --- | --- | --- |
| 1 | Dynamic ICIR gate | DoubleEnsemble is not optimal in every validation-time market state. A regime-specific blend can improve net IR. | Train DoubleEnsemble, LightGBM, and Linear; learn validation-period regime weights from top-k label mean minus turnover proxy. | Excess IR with cost > 1.935775 and at least two yearly slices do not lose to baseline. |
| 2 | Low-turnover reranker | The DoubleEnsemble signal is strong enough; retaining previous top-k names can reduce cost drag. | Add a validation-selected previous-holding score boost after the dynamic blend. | With-cost return improves more than without-cost return. |
| 3 | Residual LightGBM | DoubleEnsemble captures the main signal, but a second tree model can learn remaining cross-sectional residuals. | Fit a residual model on label minus DoubleEnsemble prediction, then validation-search the residual weight. | Rank IC and with-cost IR both improve. |
| 4 | Regime split DoubleEnsemble | Different volatility/liquidity regimes need different DoubleEnsemble parameters. | Train or select separate DoubleEnsemble parameter sets by regime. | Each selected regime has enough validation samples and the full test IR improves. |
| 5 | Cost-aware TopK proxy | MSE training is misaligned with the final top-k, costed objective. | Validate score transforms by top-k label mean minus turnover penalty. | Net IR improves without unstable parameter sensitivity. |
| 6 | Relation residual | Industry/concept relation features can add information missing from tabular DoubleEnsemble. | Train a lightweight relation residual or sector aggregate residual on top of DoubleEnsemble. | Residual has low correlation to DoubleEnsemble and improves net IR. |
| 7 | Alpha158 + Alpha360 blend | Alpha360 adds complementary signals. | Train DE-158 and DE-360, then validation-blend by ICIR or top-k objective. | Blend improves Rank IC and with-cost IR under the same strategy. |
| 8 | Walk-forward refit | Fixed training ending in 2020 is stale for 2024-2026. | Refit yearly or semi-annually using only historical data available at each prediction period. | Walk-forward net IR improves after accounting for training cost. |

## First Implementation

The first runnable candidate is `DynamicMetaEnsembleModel`.

It trains three native Qlib models:

- `DEnsembleModel`
- `LGBModel`
- `LinearModel`

It then learns:

- global validation weights
- validation-regime weights based on cross-sectional signal spread and model disagreement
- optional previous-holding score boost selected by validation top-k return minus a turnover proxy

This keeps the first pass close to the proven DoubleEnsemble baseline while adding regime awareness and cost awareness.

## Completed Local Runs

| Config | Status | IC | Rank IC | Excess annualized return with cost | Excess IR with cost | Readout |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `workflow_config_dynamic_meta_ensemble_Alpha158_2026_local.yaml` | Completed | 0.020180 | 0.023454 | 0.062487 | 0.925979 | Failed. Validation gate over-allocated to weaker sub-models and did not generalize. |
| `workflow_config_low_turnover_doubleensemble_Alpha158_2026_local.yaml` | Completed | 0.022712 | 0.022607 | 0.128700 | 1.755419 | Failed. Validation selected a small holding boost, but the test-period net IR still fell below baseline. |
| `workflow_config_residual_doubleensemble_lgb_Alpha158_2026_local.yaml` | Completed | 0.020944 | 0.021346 | 0.087621 | 1.130003 | Failed. Validation selected residual weight 0.25, but residual signal degraded test-period net IR. |
| `workflow_config_multiseed_doubleensemble_Alpha158_2026_local.yaml` | Completed | 0.022972 | 0.022937 | 0.129617 | 1.729592 | Failed. Three-seed bagging produced near-equal weights and reduced variance, but still underperformed the single-seed baseline. |
| `workflow_config_doubleensemble_rolling_Alpha158_2026_local.yaml` | Completed | 0.024439 | 0.023802 | 0.141236 | 1.848143 | Failed narrowly. Walk-forward refit improved over most add-on variants, but still did not beat the static baseline. |

Current decision: do not widen the dynamic-gate, low-turnover, residual-LGB, 3-seed bagging, or annual rolling-refit grids yet. The evidence now suggests the current baseline is structurally strong under the existing feature set, and further gains will likely require changing the information set or the target rather than adding wrappers around the same Alpha158 signal.

## Current Candidate

The next candidate is `MultiSeedDEnsembleModel`.

It keeps the full DoubleEnsemble training path unchanged and only changes the outer aggregation:

- train the same DoubleEnsemble configuration with multiple `random_state` values
- generate validation predictions for each seed
- learn simple seed weights from validation ICIR, or use equal weight as a fallback
- bag test predictions with the learned weights

This is the lowest-risk way to test whether the baseline is variance-limited rather than signal-limited.

Result: the 3-seed run produced weights close to uniform and did not beat the baseline, so the current baseline does not look variance-limited enough for simple seed bagging to unlock the missing performance.

## Next Candidate

The next candidate is walk-forward refit with the original DoubleEnsemble model.

Instead of changing the model family, it changes the training window:

- keep the same DoubleEnsemble parameters
- split the `2024-01-01` to `2026-04-30` test span into rolling blocks
- retrain before each block with only historical data
- concatenate block predictions and evaluate them under the same backtest rule

This directly tests whether the current failure mode is stale training data rather than weak model structure.

## Cost-aware Candidate

The next model line is `CostAwareDEnsembleModel`.

It does not replace the existing DoubleEnsemble baseline. It keeps the same LightGBM `mse` sub-model training, but changes two outer-layer decisions:

- `sample_reweight` adds top-k-oriented label rank and stability signals on top of the original trajectory-based score
- final sub-model weights are learned from validation `topk return - turnover penalty` instead of fixed equal weights

This is the smallest implementation that moves the training signal toward `topk + cost` without rewriting the LightGBM objective.

## Compute Plan

Use LightGBM's internal threading to consume CPU cores. Avoid running multiple heavy sub-models concurrently in the same process because DoubleEnsemble already performs repeated LightGBM training and feature shuffling.

Recommended local settings:

- `num_threads: 20` for LightGBM and DoubleEnsemble on a 20-thread machine.
- `kernels: 1` and `joblib_backend: threading` in `qlib_init` to avoid nested process contention.
- Keep `search_step` at `0.1` for the first pass. Tighten to `0.05` only after a candidate beats the baseline.
- Start with `turnover_boost_grid: [0.0, 0.01, 0.02, 0.03, 0.05]`; larger grids should wait until the first pass shows promise.

## Run

```powershell
python -m qlib.cli.run examples\benchmarks\BeatDoubleEnsemble\workflow_config_dynamic_meta_ensemble_Alpha158_2026_local.yaml
```

Low-turnover variant:

```powershell
python -m qlib.cli.run examples\benchmarks\BeatDoubleEnsemble\workflow_config_dynamic_meta_ensemble_low_turnover_Alpha158_2026_local.yaml
```

DoubleEnsemble residual LightGBM:

```powershell
python -m qlib.cli.run examples\benchmarks\BeatDoubleEnsemble\workflow_config_residual_doubleensemble_lgb_Alpha158_2026_local.yaml
```

Multi-seed DoubleEnsemble:

```powershell
python -m qlib.cli.run examples\benchmarks\BeatDoubleEnsemble\workflow_config_multiseed_doubleensemble_Alpha158_2026_local.yaml
```

Rolling DoubleEnsemble:

```powershell
python examples\benchmarks\BeatDoubleEnsemble\rolling_doubleensemble.py run
```

Cost-aware DoubleEnsemble:

```powershell
python -m qlib.cli.run examples\benchmarks\BeatDoubleEnsemble\workflow_config_cost_aware_doubleensemble_Alpha158_2026_local.yaml
```

Compare both against:

```powershell
python -m qlib.cli.run examples\benchmarks\DoubleEnsemble\workflow_config_doubleensemble_Alpha158_2026_local.yaml
```
