# Transcendence Experiment Governance

This directory is the hard-gate layer for "beat baseline" claims. Any new model must provide round artifacts, leaderboard evidence, and SOTA gate checks before a go decision.

## 1) Baseline (frozen reference)

Baseline source: `examples/benchmarks/BeatDoubleEnsemble/README.md`

- IC: `0.024251`
- RankIC: `0.023848`
- costed AnnRet: `0.147324`
- costed IR: `1.935775`

All candidate runs are compared against this baseline (or the latest accepted SOTA in `sota_snapshot.json`, whichever is stricter).

## 2) Quant gates (must pass both)

### Relative gate (vs baseline/SOTA)

- `delta_ic >= 0.000000`
- `delta_rank_ic >= 0.000000`
- `delta_costed_annret >= 0.000000`
- `delta_costed_ir >= 0.000000`

Primary decision metric: `costed IR`, tie-breaker: `costed AnnRet`.

### Absolute gate

- `ic >= 0.020000`
- `rank_ic >= 0.020000`
- `costed_annret >= 0.120000`
- `costed_ir >= 1.600000`
- `max_drawdown <= 0.250000`
- `turnover <= 0.600000`
- `runtime_sec <= 43200`
- `leakage_check == "pass"`

Any failed item is direct `NO-GO`.

## 3) Required round artifacts

Per round, submit all of:

1. `round_summary_<round_id>.md` (copy from `round_summary_template.md`)
2. New row appended to `leaderboard_template.csv` (or project leaderboard copy)
3. Updated `sota_snapshot.json` if and only if all gates pass
4. Command evidence (exact command strings and key log lines)

Minimum round fields:

- `ic`
- `rank_ic`
- `costed_annret`
- `costed_ir`
- `max_drawdown`
- `turnover`
- `runtime_sec`
- `leakage_check`
- `command`
- `status`

## 4) Command templates

Quick smoke (chain validation only, no full training):

```powershell
python -m quant_master.cli.run <workflow_config.yaml> --tag smoke
```

Full run (only after smoke pass):

```powershell
python -m quant_master.cli.run <workflow_config.yaml> --tag full
```

Baseline compare reference:

```powershell
python -m quant_master.cli.run examples\benchmarks\DoubleEnsemble\workflow_config_doubleensemble_Alpha158_2026_local.yaml
```

## 5) Status convention

- `smoke_passed`: artifact and parse checks passed; no full metric claim.
- `full_passed`: full run completed and metrics valid.
- `accepted_sota`: passed relative + absolute gates; SOTA updated.
- `rejected`: full run completed but failed one or more gates.
- `invalid`: leakage/command/artifact incompleteness.

## 6) Go / No-Go rule

`GO` requires:

- full artifact bundle complete
- smoke and full run evidence present
- all absolute gates pass
- all relative deltas non-negative

Otherwise: `NO-GO`.
