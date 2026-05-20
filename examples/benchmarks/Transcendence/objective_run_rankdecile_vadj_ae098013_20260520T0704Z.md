# Objective Run Evidence (RankDecile VolAdj)

- timestamp_utc: `2026-05-20T07:04:00Z`
- candidate_run_id: `ae0980136de44dc58a0b9d3f7d947363`
- baseline_run_id: `7406e47063e9479cb34d300b9ed03bad`
- candidate_workflow: `examples/benchmarks/Transcendence/workflow_config_regime_horizon_de_only_objective_rankdecile_vadj_Alpha158_2026_csi300.yaml`

## Label / Objective Definition

- adapter scope: build label only for `train` / `valid`; `test` has no label construction.
- mode: `rank_decile_spread`
- multi-horizon target: `[1, 5, 10]` with weights `[0.6, 0.3, 0.1]`
- target transforms:
  - market-relative (cross-sectional mean neutralization): `true`
  - instrument rolling-vol adjustment (`window=20`, `floor=1e-4`): `true`
  - decile spread boost (`decile=0.1`, `scale=0.45`)
  - clip: `[-6, 6]`
- valid objective label uses same transform family (no test-period leakage path).

## 7406 Audit (Strong vs Weak Settings)

- `7406e470`: strongest observed in this family, with `topk=45`, `turnover_penalty=5e-05`, `risk_penalty=0`, `memory_boost_grid=[0,0.005]`, `zscore_clip=100`, de-only h1.
- weaker examples:
  - `773bd6d`: de-only baseline (`topk=50`, no turnover penalty, clip=2.8) -> lower costed AnnRet/IR.
  - `0ed35c`: raw-rank exec (`topk=50`, no turnover penalty) -> clearly lower costed AnnRet/IR.
  - `2ac6`: topk40/drop3 + higher turnover penalty grid -> lower IR than 7406.
  - `bc641`: add `lgb_h5` member -> weaker than de-only 7406.
  - `6feaa`: moderate regime+rank-score+penalty-grid setup -> negative costed performance.

## Metrics (Candidate vs 7406)

| metric | candidate | 7406 | delta |
| --- | ---: | ---: | ---: |
| IC | `0.02017021428387073` | `0.023921221367623043` | `-0.0037510070837523113` |
| RankIC | `0.015728081550430508` | `0.021225039288402008` | `-0.0054969577379715` |
| costed AnnRet | `0.13866665629749725` | `0.24466463608994535` | `-0.1059979797924481` |
| costed IR | `1.5713130469238665` | `2.799983676714277` | `-1.2286706297904106` |
| max drawdown | `-0.0574967621132277` | `-0.04812598495553819` | `-0.009370777157689508` |
| turnover | `0.17506983420422315` | `0.17499578770011953` | `+0.00007404650410361424` |
| runtime_sec | `435.113` | `412.693` | `+22.42` |

## Breakthrough Result

- breakthrough_over_7406: `false`
