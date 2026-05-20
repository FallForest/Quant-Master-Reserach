# Transcendence Hybrid Model Notes

## 1) Availability and Dependency Risk (evidence-backed)

Runtime probe in this workspace (`py -3`) shows:

- `lightgbm`: available (`4.6.0`)
- `sklearn`: available (`1.8.0`)
- `scipy`: available (`1.17.1`)
- `torch`: **not installed** (`ModuleNotFoundError`)

Model availability implications:

- `DEnsemble` / `LGBModel`: available, depend on LightGBM.
- `LinearModel`: available, depends on sklearn/scipy.
- `Transformer` / `GRU` / `ALSTM` / `HIST` / `TFT`: all PyTorch-family models, blocked when `torch` is missing.

Code evidence:

- `quant_master/contrib/model/__init__.py` treats PyTorch models as optional and skips them on `ModuleNotFoundError`.
- `quant_master/contrib/model/pytorch_*.py` models import `torch` at module import time.

## 2) Implemented Hybrid Candidate

Model file:

- `quant_master/contrib/model/transcendence_hybrid.py`

Class:

- `TranscendenceHybridModel`

Main behavior:

1. Trains base learners (default: `DEnsemble + LGB + Linear`).
2. Builds validation-time rank ensemble and learns blend weights by objective:
   - `objective = ir_weight * IR + annret_weight * AnnRet`
   - IR/AnnRet measured from daily top-k validation portfolio returns.
3. Optional residual learner:
   - residual target = `label - base_blend`
   - residual model = LightGBM
   - residual weight picked on validation by same portfolio objective.
4. Optional deep branch:
   - configurable class/module (example: GRU)
   - automatic downgrade if `torch` unavailable (logs reason, continues non-deep path).

Safety and compatibility:

- `fit/predict` API compatible with existing workflows.
- strict index alignment between prediction/label/feature before scoring.
- `inf/nan` cleanup and cross-sectional fill to prevent instability.
- feature/label fetching uses separate `dataset.prepare(...feature...)` and `dataset.prepare(...label...)` calls, which works with both classic Alpha handlers and Factor Worker style handlers that still conform to `DatasetH` + `DataHandlerLP` interfaces.

## 3) Moderate Workflow (2024-2026 test)

Config:

- `examples/benchmarks/Transcendence/workflow_config_transcendence_hybrid_moderate_Alpha158_2026_csi300.yaml`

Key constraints:

- data source: `.qmData/cn_data`
- test window: `2024-01-01` ~ `2026-04-30`
- moderate runtime controls:
  - compact base learners (`de_main` with `num_models=2`, reduced epochs)
  - constrained ensemble search (`search_step=0.2`, random candidates capped)
  - bounded residual training rounds

Deep branch in this YAML is enabled intentionally for capability probing, but auto-skips when `torch` is absent.

## 4) Run / Verify

Compile check:

```powershell
py -3 -m py_compile quant_master/contrib/model/transcendence_hybrid.py
```

Model config init smoke:

```powershell
@'
import yaml
from quant_master.utils import init_instance_by_config
cfg = yaml.safe_load(open("examples/benchmarks/Transcendence/workflow_config_transcendence_hybrid_moderate_Alpha158_2026_csi300.yaml", "r", encoding="utf-8"))
init_instance_by_config(cfg["task"]["model"])
print("model_init_ok")
'@ | py -3 -
```

Full workflow:

```powershell
py -3 -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_transcendence_hybrid_moderate_Alpha158_2026_csi300.yaml
```

If using a Factor Worker handler, replace `task.dataset.kwargs.handler` (`class`, `module_path`, and handler `kwargs`) while keeping `DatasetH` segments unchanged.
