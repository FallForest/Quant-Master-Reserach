# Transcendence Data and Config Notes

## 1) Data boundary and split

- Data provider path: `.qmData/cn_data`
- Region/market: `cn` / `csi300`
- Benchmark: `SH000300`

Main config (`workflow_config_regime_horizon_cost_ensemble_Alpha158Alpha360_2026_csi300.yaml`):
- Handler boundary: `2008-01-01` to `2026-04-30`
- Train: `2008-01-01` to `2018-12-31`
- Valid: `2019-01-01` to `2023-12-31`
- Test/Backtest: `2024-01-01` to `2026-04-30`

Moderate validation-search config (`workflow_config_regime_horizon_cost_ensemble_moderate_Alpha158_2026_csi300.yaml`):
- Handler boundary: `2012-01-01` to `2026-04-30`
- Train: `2012-01-01` to `2019-12-31`
- Valid: `2020-01-01` to `2023-12-31`
- Test/Backtest: `2024-01-01` to `2026-04-30`

Quick smoke config (`workflow_config_regime_horizon_cost_ensemble_quick_smoke_Alpha158Alpha360_2024_csi300.yaml`):
- Handler boundary: `2016-01-01` to `2024-12-31`
- Train: `2018-01-01` to `2022-12-31`
- Valid: `2023-01-01` to `2023-12-31`
- Test/Backtest: `2024-01-01` to `2024-12-31`
- Stability override: DE branch uses `enable_sr: false` and `enable_fs: false`

Moderate config also keeps DE `enable_sr: false` / `enable_fs: false` to reduce runtime instability before full runs.

## 2) Model kwargs contract (currently consumed)

Current `RegimeHorizonCostEnsembleModel` consumes and this benchmark now only passes:

- `horizon_model_specs` (multi-horizon base model list)
- `topk`
- `search_step`
- `turnover_penalty` or `turnover_penalty_grid`
- `risk_penalty` or `risk_penalty_grid`
- `memory_boost_grid`
- `regime_consensus_quantiles`
- `regime_disagreement_quantiles`
- `min_regime_samples`
- `use_rank_score`, `zscore_clip`, `neutralize_daily_mean`
- `enforce_horizon_monotonic`, `monotonic_direction`
- `random_state`

Note:
- The historical "Alpha158Alpha360" filename keeps compatibility naming, but current configs here
  still run `Alpha158` handler only unless secondary handler support is implemented in model code.

For base model details, each spec uses `model_kwargs` directly, including DE/LGBM/Linear parameters.
For `DEnsembleModel` configs, keep parameter contracts aligned:
- `len(sample_ratios) == bins_fs`
- `len(sub_weights) == num_models`

## 3) Why this is still beyond baseline

Compared with CSI300 Alpha158 DoubleEnsemble baseline, this candidate is not a plain wrapper:

- Multi-horizon supervision (`h=1/5/10`) through horizon label adapter.
- Regime-aware blending by prediction consensus/disagreement bins.
- Cost-aware objective with turnover penalty and explicit risk penalty.
- Memory boost search to reduce churn in top-k holdings.
- Monotonic horizon-weight constraint as stability/risk control.

## 4) Leakage prevention checklist

- Fit window ends before test start in both configs.
- Weight search / regime thresholds / memory boost are learned on train+valid only.
- No test-label feedback into model selection.
- Horizon label smoothing is done within segment and instrument only.
- Backtest uses prediction outputs of the declared test segment.

## 5) Reserved for next phase (NOT passed into model kwargs now)

To avoid init/runtime failures and paper-only fields, the following are currently **documentation-only**:

- Alpha158 + Alpha360 dual handler fusion
- `secondary_handler`
- feature blending controls (e.g., blend mode, blend grid)
- rolling retrain/revalidation controls
- transformer branch integration

These can move into YAML kwargs only after Worker A confirms parser/usage support.

## 6) Run commands

Quick smoke (full workflow):

```bash
python -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_quick_smoke_Alpha158Alpha360_2024_csi300.yaml
```

Main run:

```bash
python -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_Alpha158Alpha360_2026_csi300.yaml
```

Moderate run:

```bash
python -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_moderate_Alpha158_2026_csi300.yaml
```

Model-only init smoke (no dataset, no training):

```bash
python - <<'PY'
import yaml
from quant_master.utils import init_instance_by_config

for fp in [
    "examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_Alpha158Alpha360_2026_csi300.yaml",
    "examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_moderate_Alpha158_2026_csi300.yaml",
    "examples/benchmarks/Transcendence/workflow_config_regime_horizon_cost_ensemble_quick_smoke_Alpha158Alpha360_2024_csi300.yaml",
]:
    cfg = yaml.safe_load(open(fp, "r", encoding="utf-8"))
    init_instance_by_config(cfg["task"]["model"])
    print("INIT_OK", fp)
PY
```

## 7) Baseline gate for promotion

- Current strongest baseline (costed): CSI300 Alpha158 DoubleEnsemble
- Test window: `2024-01-01` to `2026-04-30`
- Baseline costed IR: `1.935775`

Promotion gate:
- Costed IR on the same test window should be `> 1.935775`.
- Turnover and drawdown should be non-degraded under same backtest cost settings.
