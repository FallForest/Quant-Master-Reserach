# Amount-flow feature discovery (CSI300)

Date: 2026-07-22

## Scope

- Training screen: 2016-01-01 to 2020-12-31
- Validation screen: 2021-01-01 to 2023-12-31
- Final test/backtest: 2024-01-01 to 2026-04-30
- Baseline: registered `rhce_v1`, Alpha158, topk 3, n_drop 1, hold threshold 2
- Selection rule: same train/validation Rank IC sign, stable Rank IC >= 0.005, validation coverage >= 95%, pairwise absolute rank correlation < 0.8

The local TDX provider has `amount` but no native `vwap` file. The discovery library therefore includes an
`amount / (volume * 100)` VWAP proxy, but proxy replacement and feature additions are evaluated separately.

## Stable candidates

| Feature | Train Rank IC | Valid Rank IC | Stable Rank IC |
|---|---:|---:|---:|
| AF_RANGE_VOL20 | -0.021776 | -0.031507 | 0.021776 |
| AF_AMOUNT_REL60 | -0.019530 | -0.013950 | 0.013950 |
| AF_OVERNIGHT_INTRADAY_SPREAD | 0.019209 | 0.013592 | 0.013592 |
| AF_RET_AMOUNT_CORR60 | -0.012808 | -0.020533 | 0.012808 |
| AF_SIGNED_AMOUNT20 | -0.016077 | -0.011605 | 0.011605 |
| AF_RET_AMOUNT_CORR20 | -0.011025 | -0.015684 | 0.011025 |
| AF_SIGNED_AMOUNT5 | -0.023927 | -0.009291 | 0.009291 |
| AF_RET_AMOUNT_CORR10 | -0.009977 | -0.007493 | 0.007493 |
| AF_AMOUNT_MOM20 | -0.019777 | -0.007402 | 0.007402 |
| AF_AMOUNT_MOM5 | -0.016028 | -0.005218 | 0.005218 |

`AF_AMIHUD20`, `AF_AMIHUD60`, and `AF_VOL_REGIME5_20` changed sign between training and validation and were
rejected. The complete 21-feature scan is in `analysis/amount_flow_feature_discovery_csi300.json`.

## Controlled model results

| Variant | Feature change | IC | Rank IC | Costed excess annual return | Costed IR | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Original Alpha158 | 0.020279 | 0.012819 | 29.29% | 1.038 | -24.32% |
| Alpha168 | VWAP proxy + 10 screened features | 0.014151 | 0.014239 | -6.59% | -0.372 | -36.16% |
| Alpha160 proxy | VWAP proxy + range/overnight core | 0.017149 | 0.016139 | 11.80% | 0.623 | -13.18% |
| Alpha158 proxy | VWAP proxy only | 0.013418 | 0.015724 | -19.73% | -1.043 | -50.71% |
| Alpha160 baseline | Original Alpha158 + range/overnight core | 0.013454 | 0.012576 | -7.83% | -0.445 | -38.88% |

Run IDs:

- Baseline: `e612e7a68aa64b3eaec266384321dbea`
- Alpha168: `36f6035908754a6aa53d3d01e3b1c8bb`
- Alpha160 proxy: `6bd83f860e444c71878297d65c8a3d49`
- Alpha158 proxy: `77b7dbc73aec4c5eb9cfe0d0e051860f`
- Alpha160 baseline: `005b4585bc654a7eb4d72c1679ece053`

## Decision

NO-GO for production registration. The two core features materially rescue the corrected-VWAP model, and they
rank highly in LightGBM gain (`AF_OVERNIGHT_INTRADAY_SPREAD` is third for h1; `AF_RANGE_VOL20` is fifth for h5),
but they do not improve the untouched production baseline. The result also shows that filling Alpha158's currently
empty VWAP0 column is not automatically beneficial for this shallow RHCE configuration.

Keep the feature library and workflows for future ablations. Do not switch `ui/server/model_registry.json` away
from `rhce_v1` based on these runs.
