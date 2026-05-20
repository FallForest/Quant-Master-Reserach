# Meta-Label Moderate Run (Alpha158 / csi300)

- provider: `.qmData/cn_data`
- train: `2016-01-01` to `2020-12-31`
- valid: `2021-01-01` to `2023-12-31`
- test: `2024-01-01` to `2026-04-30`
- SOTA reference run: `7406e47063e9479cb34d300b9ed03bad` (costed IR `3.0230019401859436`)

## Label definitions

- `top_bottom`: per day cross-sectional rank-percentile on forward return; `+1` if `rank_pct >= 0.9`, `-1` if `rank_pct <= 0.1`, else `0`.
- `rank`: per day cross-sectional continuous target `2 * (rank_pct - 0.5)` in `[-1, 1]`.

## Workflow configs

- `examples/benchmarks/Transcendence/workflow_config_transcendence_metalabel_topbottom_moderate_Alpha158_2026_csi300.yaml`
- `examples/benchmarks/Transcendence/workflow_config_transcendence_metalabel_rank_moderate_Alpha158_2026_csi300.yaml`

## Results

| variant | run_id | IC | Rank IC | costed_annret | costed_IR | max_drawdown | turnover | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| top_bottom | `29864d9c5d00463b9fdbc065c10b0093` | 0.011667 | 0.032175 | 0.117154 | 1.662795 | -0.058895 | 0.200285 | 199.980 |
| rank | `4a98f99bdb6848bab789ff6c46d0a1ff` | 0.012182 | 0.031144 | 0.048172 | 0.559719 | -0.104495 | 0.198090 | 192.306 |

## Conclusion

- Best variant: `top_bottom` (costed IR `1.662795`).
- Breakthrough vs `7406` model SOTA: **No**.
