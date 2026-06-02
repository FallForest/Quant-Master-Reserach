# Expanded Factor Signal Library Summary (20260522T114855Z)

- base_run_id: `7406e47063e9479cb34d300b9ed03bad`
- test_period: `2024-01-01` to `2026-04-30`
- hard_gate: `IR > 2.9` and `AnnRet > 0.27`
- passes_hard_gate: `no`

## Top Candidates

| signal | family | variant | RankIC_IR | IC_IR | bucket_IR | best_topk | best_n_drop | costed_IR | AnnRet | maxDD | turnover |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_equal_top5 | simple_combo | rank_blend | 1.891955 | 0.523675 | -0.418434 | 55 | 2 | 0.797366 | 0.059700 | -0.072178 | 0.071781 |
| vp_div_10__raw | volume_price_divergence | raw | 1.682009 | 0.845014 | 0.263966 | 50 | 2 | 0.588620 | 0.058611 | -0.110288 | 0.076986 |
| vp_div_10__rank | volume_price_divergence | rank | 1.682009 | 0.507053 | 0.263966 | 50 | 2 | 0.588620 | 0.058611 | -0.110288 | 0.076986 |
| vp_div_10__robustz | volume_price_divergence | robustz | 1.681222 | 0.611638 | 0.263966 | 50 | 2 | 0.586940 | 0.058445 | -0.110324 | 0.076981 |
| vp_div_10__mkt_neutral_robustz | volume_price_divergence | mkt_neutral_robustz | 1.681222 | 0.611638 | 0.263966 | 50 | 2 | 0.586940 | 0.058445 | -0.110324 | 0.076981 |
| vol_comp_10_60__rank | volatility_compression_expansion | rank | 1.459847 | 0.196099 | -0.199258 | 55 | 2 | 0.550382 | 0.042089 | -0.096367 | 0.072974 |
| vol_comp_10_60__raw | volatility_compression_expansion | raw | 1.459847 | 0.165397 | -0.199258 | 55 | 2 | 0.550382 | 0.042089 | -0.096367 | 0.072974 |
| vol_comp_10_60__mkt_neutral_robustz | volatility_compression_expansion | mkt_neutral_robustz | 1.459834 | 0.166051 | -0.199258 | 55 | 2 | 0.550382 | 0.042089 | -0.096367 | 0.072974 |
| vol_comp_10_60__robustz | volatility_compression_expansion | robustz | 1.459834 | 0.166051 | -0.199258 | 55 | 2 | 0.550382 | 0.042089 | -0.096367 | 0.072974 |
| combo_equal_top3 | simple_combo | rank_blend | 1.682009 | 0.506787 | 0.263966 | 45 | 2 | 0.535627 | 0.055916 | -0.102811 | 0.087927 |
| combo_icir_weighted_top4 | simple_combo | rank_blend | 1.682009 | 0.506655 | 0.263966 | 45 | 2 | 0.535627 | 0.055916 | -0.102811 | 0.087927 |
| vp_div_20__robustz | volume_price_divergence | robustz | 1.698330 | 0.486425 | -0.593545 | 55 | 4 | 0.300476 | 0.020859 | -0.099045 | 0.144314 |
| vp_div_20__mkt_neutral_robustz | volume_price_divergence | mkt_neutral_robustz | 1.698330 | 0.486425 | -0.593545 | 55 | 4 | 0.300476 | 0.020859 | -0.099045 | 0.144314 |
| vp_div_20__raw | volume_price_divergence | raw | 1.698172 | 0.545049 | -0.593545 | 55 | 4 | 0.298954 | 0.020756 | -0.099045 | 0.144314 |
| vp_div_20__rank | volume_price_divergence | rank | 1.698172 | 0.210285 | -0.593545 | 55 | 4 | 0.298954 | 0.020756 | -0.099045 | 0.144314 |

## Leakage Notes

- All rolling features are instrument-wise backward-looking only.
- No signal uses future label or forward-filled future values.
- Label alignment follows existing workflow label index directly.