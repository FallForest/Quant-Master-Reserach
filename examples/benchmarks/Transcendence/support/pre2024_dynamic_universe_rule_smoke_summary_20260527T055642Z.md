# pre-2024 dynamic universe rule smoke

- task_id: Q-DYNAMIC-UNIVERSE-PRE2024-RULE-SMOKE
- status: completed
- verdict: NO_GO
- is_proxy: `True`
- gate_pass_count: `0` / `4`
- data_window: `2020-01-01..2023-12-31`
- uses_2024_plus: `False`
- costs: open `0.0005`, close `0.0015`

## Top rules
- liq_adj_mom20_amt60: IR=0.3051, AnnRet=0.0474, MDD=-0.5626, TO=0.2947, gate=False, reasons=year_ir_not_positive=2022,2023;year_ir_gt_1_count=1<3;combined_ir=0.305072<1.8;combined_mdd=-0.562584<-0.12
- vwap_quality_mom10: IR=-0.6262, AnnRet=-0.1960, MDD=-0.7713, TO=0.4056, gate=False, reasons=year_ir_not_positive=2021,2022,2023;year_ir_gt_1_count=0<3;combined_ir=-0.626227<1.8;combined_mdd=-0.771316<-0.12
- short_reversal5_liq_stable: IR=-0.7342, AnnRet=-0.2257, MDD=-0.7700, TO=0.3626, gate=False, reasons=year_ir_not_positive=2021,2022,2023;year_ir_gt_1_count=0<3;combined_ir=-0.734185<1.8;combined_mdd=-0.770032<-0.12
- amount_trend_volume_stability: IR=-0.8357, AnnRet=-0.2130, MDD=-0.7657, TO=0.5100, gate=False, reasons=year_ir_not_positive=2021,2022,2023;year_ir_gt_1_count=0<3;combined_ir=-0.83566<1.8;combined_mdd=-0.765749<-0.12

## Artifacts
- summary_json: C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\support\pre2024_dynamic_universe_rule_smoke_summary_20260527T055642Z.json
- summary_md: C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\support\pre2024_dynamic_universe_rule_smoke_summary_20260527T055642Z.md
- rules_csv: C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\support\pre2024_dynamic_universe_rule_smoke_rules_20260527T055642Z.csv
- year_metrics_csv: C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\support\pre2024_dynamic_universe_rule_smoke_year_metrics_20260527T055642Z.csv
- universe_csv: C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\support\pre2024_dynamic_universe_rule_smoke_universe_20260527T055642Z.csv
