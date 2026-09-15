# Daily market-microstructure proxy research (CSI300)

Date: 2026-07-22

## Scope

- Raw data: unified TDX daily provider at `~/.quant_master/quant_master_data/tdx_cn_data`
- Training screen: 2016-01-01 to 2020-12-31
- Validation screen: 2021-01-01 to 2023-12-31
- Final test/backtest: 2024-01-01 to 2026-04-30
- Baseline: registered `rhce_v1`, Alpha158, topk 3, n_drop 1, hold threshold 2
- Selection rule: same train/validation Rank IC sign, stable Rank IC >= 0.005,
  validation coverage >= 95%, pairwise absolute rank correlation < 0.8

The features are daily proxies. They do not claim to reconstruct a true order book, aggressor side, or exchange
price-limit state. The fixed +/-10% distance features are reference distances only and do not model ST, ChiNext,
STAR, or Beijing Stock Exchange limit regimes.

## Stable low-redundancy candidates

| Feature | Family | Train Rank IC | Valid Rank IC | Stable Rank IC |
|---|---|---:|---:|---:|
| MSP_IDIO_VOL20 | idiosyncratic volatility | -0.021442 | -0.037146 | 0.021442 |
| MSP_CORWIN_SCHULTZ20 | high-low spread proxy | -0.014037 | -0.034110 | 0.014037 |
| MSP_UPLIMIT10_DIST | +10% reference distance | 0.008105 | 0.023980 | 0.008105 |
| MSP_ROLL_SPREAD20 | serial-covariance spread proxy | -0.007906 | -0.012222 | 0.007906 |
| MSP_DOWNLIMIT10_DIST | -10% reference distance | 0.006534 | 0.010681 | 0.006534 |

Parkinson, Garman-Klass, and Rogers-Satchell volatility were individually stable but rejected by the correlation
gate after idiosyncratic volatility was selected. Return autocorrelation, zero-return ratio, and jump share did not
meet the stable Rank IC threshold. Market beta changed sign between training and validation.

The complete 12-feature scan is in `analysis/market_microstructure_feature_discovery_csi300.json`.

## Controlled model result

| Variant | Features | IC | Rank IC | Costed excess annual return | Costed IR | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| Production baseline | Alpha158 | 0.020279 | 0.012819 | 29.29% | 1.038 | -24.32% |
| Market proxy core | Alpha158 + selected 5 | 0.015777 | 0.015306 | -9.56% | -0.532 | -55.39% |

Run IDs:

- Production baseline: `e612e7a68aa64b3eaec266384321dbea`
- Market proxy core: `344968109742421185bf313dd9e01d56`

## Decision

NO-GO for production registration. The selected proxies improve out-of-sample Rank IC, but the current top-3
portfolio does not convert the stronger cross-sectional ordering into costed returns. Keep the feature registry,
handler, screening output, and research workflows for family ablations and broader portfolio experiments. Do not
switch `ui/server/model_registry.json` away from `rhce_v1` based on this run.
