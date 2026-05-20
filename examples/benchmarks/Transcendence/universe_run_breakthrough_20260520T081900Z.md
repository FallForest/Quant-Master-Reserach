# Universe/Handler Breakthrough Attempt (2026-05-20)

## Scope

- Strong architecture fixed to 7406 DE-only regime-horizon setup.
- Alternative data choices tried:
  - `Alpha360 + csi300` (comparable to 7406)
  - `Alpha158 + csi500` (different universe/benchmark)
- Test window fixed: `2024-01-01` to `2026-04-30`.

## Available Universes (local audit)

- instrument files: `csi300`, `csi500`, `csi800`, `all` (plus csi100/csi1000/csiall)
- active symbols on `2026-04-30`:
  - `csi300`: 300
  - `csi500`: 500
  - `csi800`: 800
  - `all`: 5517
- benchmark symbols in local instruments (`all.txt`): `SH000300`, `SH000905` both present.

## Configs Created

- `examples/benchmarks/Transcendence/workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_universe_Alpha360_2026_csi300.yaml`
- `examples/benchmarks/Transcendence/workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_universe_Alpha158_2026_csi500.yaml`

## Runs and Metrics

Baseline (for comparable check):
- run_id: `7406e47063e9479cb34d300b9ed03bad`
- setup: `Alpha158 + csi300 + SH000300`
- IC `0.023921`, RankIC `0.021225`, costed AnnRet `0.244665`, costed IR `2.799984`

New runs:
1. `Alpha360 + csi300 + SH000300` (comparable)
   - run_id: `54bd583a84d54909a9d572af9426ff95`
   - IC `0.017211`, RankIC `0.011347`, costed AnnRet `0.186073`, costed IR `1.556170`
   - verdict vs 7406: **not breakthrough**

2. `Alpha158 + csi500 + SH000905` (not directly comparable)
   - run_id: `a60be2d0ec60405ebad3a7c51b68c54a`
   - IC `0.022273`, RankIC `0.018122`, costed AnnRet `0.060085`, costed IR `0.819982`
   - note: different benchmark/universe; indicates weaker portfolio strength under this setting.

## Final

- Comparable breakthrough (`Alpha360+csi300` beating 7406): **No**.
- Fairness note: `csi500` result is informative for cross-universe robustness but not apples-to-apples against `7406` csi300 baseline.
