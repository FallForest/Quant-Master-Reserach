# Transcendence AlphaExt DEnsemble Moderate Run Evidence

- run_id: `05ef8bd12e0e407f9fdf0cad3ef72652`
- workflow_config: `examples/benchmarks/Transcendence/workflow_config_transcendence_factor_densemble_moderate_AlphaExt_2026_csi300.yaml`
- timestamp_utc: `20260520T055012Z`

## Candidate metrics (default topk=50 n_drop=5)

- IC: `0.021974775039`
- RankIC: `0.021724444842`
- costed AnnRet: `0.145053351945`
- costed IR: `1.374036665291`
- maxDD: `-0.092214221233`
- turnover: `0.195845124202`
- runtime_sec: `442.753`

## Comparison (signal default组合)

- vs 7406 (IR=2.799983676714277 AnnRet=0.24466463608994535):
  - delta IC: `-0.001946446328`
  - delta RankIC: `0.000499405554`
  - delta costed AnnRet: `-0.099611284144`
  - delta costed IR: `-1.425947011423`
- vs original baseline (IR=1.935775 AnnRet=0.147324):
  - delta IC: `-0.002276224961`
  - delta RankIC: `-0.002123555158`
  - delta costed AnnRet: `-0.002270648055`
  - delta costed IR: `-0.561738334709`

## Portfolio scan best (topk/n_drop small scan)

- source: `examples/benchmarks/Transcendence/portfolio_scan_alphaext_densemble_moderate_05ef8bd1_summary_05ef8bd1_20260520T055023Z.json`
- best: `topk=54, n_drop=1, open_cost=0.0005, close_cost=0.0015`
- best costed AnnRet: `0.190747712206`
- best costed IR: `2.233250563406`
- best maxDD: `-0.067390513693`
- best turnover: `0.035775172481`
- delta vs default组合:
  - costed AnnRet: `+0.045694360260`
  - costed IR: `+0.859213898115`
- delta vs 7406:
  - costed AnnRet: `-0.053916923884`
  - costed IR: `-0.566733113308`

## Verdict

- beats 7406 (default组合): `False`
- beats original baseline (default组合): `False`
- portfolio_scan_best_beats_7406: `False`
