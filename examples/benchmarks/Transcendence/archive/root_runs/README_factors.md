# Transcendence Factor Library (Expanded)

## Handler and workflow

- Handler: `quant_master.contrib.data.transcendence_handler.TranscendenceAlpha`
- Smoke workflow: `examples/benchmarks/Transcendence/workflow_config_transcendence_factor_smoke_AlphaExt_2024_csi300.yaml`
- Audit artifacts:
  - `examples/benchmarks/Transcendence/factor_audit_transcendence_smoke_20260520.json`
  - `examples/benchmarks/Transcendence/factor_audit_transcendence_smoke_20260520.csv`

## Factor families

The handler keeps an expanded Alpha158 base set, then appends Transcendence factors:

1. Multi-period momentum / reversal
   `TX_MOM_*`, `TX_REV_*`, `TX_VWAP_MOM_*`, `TX_MOM_ACCEL_*`, `TX_MOM_SPREAD_*`
2. Volume-price divergence
   `TX_PV_DIV_*`, `TX_PV_CORR_*`, `TX_PV_COV_*`, `TX_PV_LAGCORR_*`, `TX_PV_IMBAL_*`
3. Volatility compression / expansion
   `TX_RVOL_*`, `TX_RANGE_VOL_*`, `TX_AMP_*`, `TX_VOL_REGIME_*`, `TX_AMP_REGIME_*`
4. Price-volume correlation and liquidity
   `TX_RET_VOL_CORR_*`, `TX_ABSRET_VRET_CORR_*`, `TX_AMIHUD_*`, `TX_DVOL_Z_*`, `TX_TURN_Z_*`
5. Skewness / tail risk
   `TX_SKEW_*`, `TX_KURT_*`, `TX_DOWNSIDE_*`, `TX_TAIL_Q05_*`, `TX_TAIL_Q95_*`, `TX_LEFT_TAIL_GAP_*`
6. Market-relative strength (benchmark-aware)
   `TX_EXCESS_RET_*`, `TX_REL_MOM_*`, `TX_BETA_*`, `TX_IDIO_VOL_*`

## Smoke command

```bash
py - <<'PY'
from pathlib import Path
from ruamel.yaml import YAML
import quant_master
from quant_master.utils import init_instance_by_config

cfg_path = Path("examples/benchmarks/Transcendence/workflow_config_transcendence_factor_smoke_AlphaExt_2024_csi300.yaml")
cfg = YAML(typ="safe", pure=True).load(cfg_path.read_text(encoding="utf-8"))
quant_master.init(**cfg["quant_master_init"])
dataset = init_instance_by_config(cfg["task"]["dataset"])
for seg in ["train", "valid", "test"]:
    df = dataset.prepare(seg, col_set="feature")
    print(seg, df.shape)
PY
```

## Smoke evidence (2026-05-20)

- Feature count: `368`
- Non-null ratio:
  - train: `1.0000`
  - valid: `1.0000`
  - test: `1.0000`
- Date coverage:
  - train: `2020-01-02` to `2022-12-30`
  - valid: `2023-01-03` to `2023-12-29`
  - test: `2024-01-02` to `2024-12-31`

## Leakage boundary

- Labels use `Ref($close, -2)/Ref($close, -1)-1` only.
- Features use current/past values only (`Ref(..., positive_window)`, rolling windows).
- No pandas shift on future timestamps, no forward-fill from future rows in custom expressions.
- Market-relative factors use same-date benchmark observations via `ChangeInstrument`.

## Moderate run evidence (2026-05-20)

- workflow: `examples/benchmarks/Transcendence/workflow_config_transcendence_factor_moderate_AlphaExt_2026_csi300.yaml`
- run_id: `d4526da7854245af954fc99cf02963f0`
- full runtime: `516.816s`
- candidate:
  - IC: `0.023280032481`
  - RankIC: `0.020276466301`
  - costed AnnRet: `0.135502681101`
  - costed IR: `1.313621400853`
  - maxDD: `-0.112773139718`
  - turnover: `0.195866134997`
- vs 7406 SOTA (`IR=2.799983676714277`, `AnnRet=0.24466463608994535`): not exceeded
- vs original baseline (`IR=1.935775`, `AnnRet=0.147324`): not exceeded
- evidence bundle:
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_moderate_20260520T053545Z.json`
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_moderate_20260520T053545Z.csv`
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_moderate_20260520T053545Z.md`

## DEnsemble moderate run evidence (2026-05-20)

- workflow: `examples/benchmarks/Transcendence/workflow_config_transcendence_factor_densemble_moderate_AlphaExt_2026_csi300.yaml`
- run_id: `05ef8bd12e0e407f9fdf0cad3ef72652`
- full runtime: `442.753s`
- default组合 (`topk=50 n_drop=5`):
  - IC: `0.021974775039`
  - RankIC: `0.021724444842`
  - costed AnnRet: `0.145053351945`
  - costed IR: `1.374036665291`
  - maxDD: `-0.092214221233`
  - turnover: `0.195845124202`
- 小扫描最优组合 (`topk=54 n_drop=1`):
  - costed AnnRet: `0.190747712206`
  - costed IR: `2.233250563406`
  - maxDD: `-0.067390513693`
  - turnover: `0.035775172481`
- vs 7406 (`IR=2.799983676714277`, `AnnRet=0.24466463608994535`): default与扫描最优均未超越
- vs original baseline (`IR=1.935775`, `AnnRet=0.147324`): default未超越
- evidence bundle:
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_densemble_moderate_20260520T055012Z.json`
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_densemble_moderate_20260520T055012Z.csv`
  - `examples/benchmarks/Transcendence/factor_model_run_transcendence_alphaext_densemble_moderate_20260520T055012Z.md`
  - `examples/benchmarks/Transcendence/portfolio_scan_alphaext_densemble_moderate_05ef8bd1_summary_05ef8bd1_20260520T055023Z.json`
  - `examples/benchmarks/Transcendence/portfolio_scan_alphaext_densemble_moderate_05ef8bd1_05ef8bd1_20260520T055023Z.csv`
