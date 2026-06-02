# Factor Augmented Meta Ensemble (20260522T120515Z)

## Protocol
- Locked-forward quarterly meta (blocked CV on prior data only).
- Sparse ridge stacking with anchor prior + correlation penalty + turnover penalty.
- Costs: `open=0.0005`, `close=0.0015`.

## Members
| key | source | anchor | expanded_factor |
|---|---|---:|---:|
| ml_7406e470 | mlruns_anchor | 1 | 0 |
| expanded_factor | expanded_factor_best | 0 | 1 |
| workerA_deep_rank | gpu_deep_stack | 0 | 0 |
| ml_d4526da7 | mlruns | 0 | 0 |
| ml_bc641cef | mlruns | 0 | 0 |
| ml_4a98f99b | mlruns | 0 | 0 |
| ml_0d7d238a | mlruns | 0 | 0 |
| ml_587bba62 | mlruns | 0 | 0 |
| ml_94a52e59 | mlruns | 0 | 0 |
| ml_5ae326c0 | mlruns | 0 | 0 |
| ml_c40b1997 | mlruns | 0 | 0 |
| ml_8df189f4 | mlruns | 0 | 0 |

## Full Test Metrics
- Meta: IR=2.820878, AnnRet=0.338626, MaxDD=-0.040894, Turnover=0.252251
- Anchor: IR=3.023002, AnnRet=0.387854, MaxDD=-0.047723, Turnover=0.254714
- Hard gate pass (`IR>2.9`, `AnnRet>0.27`): `False`

## Slice Metrics
| split | IR | AnnRet | MaxDD | Turnover |
|---|---:|---:|---:|---:|
| 2024 | 2.820878 | 0.338626 | -0.040894 | 0.252251 |
| 2025 | 0.965861 | 0.086460 | -0.037766 | 0.234562 |
| 2026_ytd | 0.574626 | 0.049167 | -0.041347 | 0.200727 |