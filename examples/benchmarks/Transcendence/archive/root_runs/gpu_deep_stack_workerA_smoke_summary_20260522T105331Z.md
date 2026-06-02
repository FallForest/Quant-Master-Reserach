# GPU Deep Stack Summary (smoke, 20260522T105331Z)

- command: `examples/benchmarks/Transcendence/gpu_deep_stack_model.py --mode smoke --tracking-uri file:./mlruns --base-run-id 7406e47063e9479cb34d300b9ed03bad --start-date 2024-01-01 --end-date 2026-04-30 --open-cost 0.0005 --close-cost 0.0015 --topk 45 --n-drop 4 --output-prefix gpu_deep_stack_workerA`
- torch: `2.12.0+cu130`; cuda_available: `True`; device: `cuda`
- test_period: `2024-01-01..2026-04-30`
- costs: `open=0.0005`, `close=0.0015`

## Best Candidate

- name: `blend_base70_deep30`
- full IR: `2.625612`
- full AnnRet: `0.227525`
- full MDD: `-0.059171`
- turnover: `0.173716`
- 2026YTD IR: `0.957658`
- 2026YTD AnnRet: `0.100927`
- hard_gate_passed: `False`

## Leakage Risk

- level: `medium`
- Walk-forward uses only historical labels per fold; no future-label fitting inside each fold.
- Candidate family is predefined (deep raw + fixed blends), but best-of-candidates is still selected on full 2024-01-01..2026-04-30 report.
- Warmup gaps are filled with base rank signal; this reduces instability but weakens pure deep attribution.