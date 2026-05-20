# Signal Portfolio Conversion Summary (20260520T105046Z)

- base_run_id: `7406e47063e9479cb34d300b9ed03bad`
- threshold: IR>2.799984 and AnnRet>=0.244665
- combo_evals: `1536`

## Best Conversion Per Signal

| signal | scenario | transform | blend | family | rebalance | topk | n_drop | hold_topk | IR | AnnRet | MDD | Turnover | breakthrough |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gru_bcbecf55 | blend | smooth3 | blend_7406_70_30 | buffered_weight | weekly | 55 | 0 | 85 | 4.697086 | 0.491320 | -0.024997 | 0.241437 | yes |
| metalabel_top_bottom_29864 | blend | raw | blend_7406_80_20 | buffered_weight | weekly | 55 | 0 | 75 | 3.256304 | 0.414900 | -0.035672 | 0.273747 | yes |
| metalabel_rank_4a98 | blend | inverted | blend_7406_70_30 | buffered_weight | weekly | 55 | 0 | 75 | 2.988539 | 0.391091 | -0.058510 | 0.266724 | yes |

## Year Slices

### gru_bcbecf55

| slice | IR | AnnRet | MDD | Turnover |
|---|---:|---:|---:|---:|
| 2024 | 4.697086 | 0.491320 | -0.024997 | 0.241437 |
| 2025 | 12.352602 | 1.948405 | -0.002699 | 0.261420 |
| 2026_ytd | -0.677655 | -0.061585 | -0.062723 | 0.199364 |
| 2024_2026_full | 4.697086 | 0.491320 | -0.024997 | 0.241437 |

- diagnostics: nonnull=1.000, spread(p90-p10)=0.631718, lag1=0.6144, rank_corr_vs_7406=0.9362

### metalabel_top_bottom_29864

| slice | IR | AnnRet | MDD | Turnover |
|---|---:|---:|---:|---:|
| 2024 | 3.256304 | 0.414900 | -0.035672 | 0.273747 |
| 2025 | 2.109614 | 0.179874 | -0.031532 | 0.257472 |
| 2026_ytd | -0.059467 | -0.005355 | -0.043787 | 0.222793 |
| 2024_2026_full | 3.256304 | 0.414900 | -0.035672 | 0.273747 |

- diagnostics: nonnull=1.000, spread(p90-p10)=0.723731, lag1=0.5401, rank_corr_vs_7406=0.9832

### metalabel_rank_4a98

| slice | IR | AnnRet | MDD | Turnover |
|---|---:|---:|---:|---:|
| 2024 | 2.988539 | 0.391091 | -0.058510 | 0.266724 |
| 2025 | 0.489176 | 0.050776 | -0.081425 | 0.250288 |
| 2026_ytd | 0.379714 | 0.037723 | -0.046213 | 0.215752 |
| 2024_2026_full | 2.988539 | 0.391091 | -0.058510 | 0.266724 |

- diagnostics: nonnull=1.000, spread(p90-p10)=0.429688, lag1=0.4569, rank_corr_vs_7406=0.9182
