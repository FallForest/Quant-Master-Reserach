# Long History Retrain Summary (20260522T134241Z)

- provider_uri: `C:\Users\15728\Desktop\Quant-Master-Research\.qmData\cn_data`
- market: `csi300`
- train/valid/test: `2020-01-01..2022-12-31` / `2023-01-01..2023-12-31` / `2024-01-01..2026-04-30`
- backtest costs: `open=0.0005` `close=0.0015`
- hard gate (non-test valid): `IR > 2.9` and `AnnRet > 0.27` => `FAIL`

## Best Non-Test (Valid 2023)

- topk/n_drop: `45/2`
- IR: `1.926143`
- AnnRet: `0.218473`
- MaxDD: `-0.075651`
- Turnover: `0.094963`

## Test (2024-01-01..2026-04-30)

- topk/n_drop: `45/2`
- IR: `1.589355`
- AnnRet: `0.207454`
- MaxDD: `-0.060140`
- Turnover: `0.094053`

## Coverage

- instruments: `405`
- calendar range: `2000-01-04..2026-05-19`
- coverage csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\long_history_retrain_coverage_20260522T134241Z.csv`