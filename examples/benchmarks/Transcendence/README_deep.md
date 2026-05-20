# Transcendence Deep/Sequence Audit (2026-05-20)

## Scope

- Target: verify whether a deep/sequence model can beat current portfolio SOTA (`IR=3.023001940185944`, `AnnRet=0.38785441547152355`).
- Data and test window: `.qmData/cn_data`, `test=[2024-01-01, 2026-04-30]`.
- Candidate config: `workflow_config_gru_deep_candidate_Alpha158_2026_csi300.yaml`.

## Environment Evidence

- `nvidia-smi` is available (RTX 3070 Laptop GPU, driver 595.97, CUDA 13.2 reported by NVIDIA-SMI).
- Current Python runtime cannot import PyTorch:
  - `ModuleNotFoundError: No module named 'torch'`
  - deep model module imports (`pytorch_gru_ts`, `pytorch_transformer_ts`) fail for the same reason.
- `.venv/Lib/site-packages/torch` exists but `torch/__init__.py` is missing, so the package is not usable by the active interpreter.

## Run Evidence

- Attempted command:
  - `py -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_gru_deep_candidate_Alpha158_2026_csi300.yaml`
- Produced run:
  - `experiment_id=984329077332834218`
  - `run_id=e3f8e220da2745f6be045b5efe35a088`
  - status `4` (failed)
  - failure: `ModuleNotFoundError: No module named 'torch'`

## Current Conclusion

- Deep direction is **currently blocked by environment**, not by model quality evidence.
- No valid deep metrics were produced in this audit, so no claim can be made about beating SOTA.

## Repro Steps After Environment Fix

1. `py -m pip install --upgrade pip`
2. `py -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision`
3. `py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"`
4. `py -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_gru_deep_candidate_Alpha158_2026_csi300.yaml`

Detailed machine-readable audit:

- `deep_audit_20260520T050053Z.json`

---

## Follow-up Audit (2026-05-20, Torch fixed + rerun)

### Environment Evidence (Python 3.12)

- `py -3.12 -m pip --version`
  - `pip 26.1 ... (python 3.12)`
- `py -3.12 -m pip install torch --upgrade`
  - installed `torch-2.12.0` successfully.
- `py -3.12 -m pip show torch`
  - version `2.12.0`, location under `Python312\\Lib\\site-packages`.
- `py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  - output: `2.12.0+cpu False`.

### Run Evidence

- Command:
  - `py -3.12 -m quant_master.cli.run examples/benchmarks/Transcendence/workflow_config_gru_deep_candidate_Alpha158_2026_csi300.yaml`
- Result:
  - run completed (not blocked by torch anymore).
  - recorder `run_id=bcbecf55a3924357ba93fc55b1140e99`
  - runtime from metrics extraction: `1123.745s` (~18.73 min), within 45-minute budget.

### Extracted Metrics

Using:

- `py -3.12 examples/benchmarks/Transcendence/extract_metrics.py --tracking-uri file:./mlruns --experiment-name workflow --format json --model-name GRU_deep_candidate --workflow-config examples/benchmarks/Transcendence/workflow_config_gru_deep_candidate_Alpha158_2026_csi300.yaml`

Got:

- `IC=0.010072499523340108`
- `Rank IC=0.036224315265373876`
- `costed_annret=-0.007653629711696119`
- `costed_ir=-0.07976613922902091`
- `max_drawdown=-0.15111771391998838`
- `turnover=0.19996371770391752`

### SOTA Comparison

- Current model SOTA signal-level reference (legacy note): `IR=2.79998`.
- Current strategy-scan SOTA (official snapshot/verification): `IR=3.023001940185944`, `AnnRet=0.38785441547152355`.
- Deep GRU candidate gap vs strategy-scan SOTA:
  - `delta_ir=-3.102768079414965`
  - `delta_annret=-0.3955080451832197`

Conclusion: deep GRU is runnable now, but this candidate is far below current SOTA and not worth immediate promotion without major redesign/tuning.
