# Transcendence AlphaExt Moderate Run Evidence

- run_id: `d4526da7854245af954fc99cf02963f0`
- workflow_config: `examples/benchmarks/Transcendence/workflow_config_transcendence_factor_moderate_AlphaExt_2026_csi300.yaml`
- timestamp_utc: `20260520T053545Z`

## Candidate metrics

- IC: `0.023280032481`
- RankIC: `0.020276466301`
- costed AnnRet: `0.135502681101`
- costed IR: `1.313621400853`
- maxDD: `-0.112773139718`
- turnover: `0.195866134997`
- runtime_sec: `516.816`

## Comparison

- vs 7406 (IR=2.799983676714277 AnnRet=0.24466463608994535):
  - delta IC: `-0.000641188887`
  - delta RankIC: `-0.000948572988`
  - delta costed AnnRet: `-0.109161954989`
  - delta costed IR: `-1.486362275861`
- vs original baseline (IR=1.935775 AnnRet=0.147324):
  - delta IC: `-0.000970967519`
  - delta RankIC: `-0.003571533699`
  - delta costed AnnRet: `-0.011821318899`
  - delta costed IR: `-0.622153599147`

## Verdict

- beats 7406: `False`
- beats original baseline: `False`
