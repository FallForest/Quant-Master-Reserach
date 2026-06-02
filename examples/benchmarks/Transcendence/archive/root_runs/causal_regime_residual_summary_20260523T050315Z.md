# Causal Regime Residual Model 20260523T050315Z

Verdict: **NO_GO**

## Protocol
- Strict quarterly past-only selection.
- Daily stress gate uses previous-day diagnostics only.
- No runtime sell wrapper or execution-layer patch is applied in this script.

## Key Metrics
- Eval window: 2025-01-01..2026-04-28
- Candidate backtest: {"ok": false, "metrics": null, "error": {"type": "ValueError", "message": "only have 164539.64975975643 SZ002311, require 164540.14808790767"}}
- Factor-meta same-window backtest: {"ok": true, "metrics": {"annret": 0.08646049377076649, "ir": 0.9658612935833345, "max_drawdown": -0.037766443596712815, "turnover": 0.23456178698737185, "elapsed_sec": 2.2526421999991726}, "error": null}
- Candidate proxy objective: 4.574376
- Factor-meta proxy objective: 4.647528
- Proxy objective delta: -0.073152

## Comparison
- factor_augmented_meta published full: {"annret": 0.3386261002570873, "ir": 2.8208783790760705, "max_drawdown": -0.04089447967616651, "turnover": 0.2522513294744337, "elapsed_sec": 7.671771799999988}
- long_history_second_order verdict: NO_GO

## Artifacts
- summary_json: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_summary_20260523T050315Z.json`
- summary_md: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_summary_20260523T050315Z.md`
- candidate_pred_pkl: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_candidate_pred_20260523T050315Z.pkl`
- candidate_pred_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_candidate_pred_20260523T050315Z.csv`
- periods_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_periods_20260523T050315Z.csv`
- selector_diag_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_selector_diag_20260523T050315Z.csv`
- slices_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_slices_20260523T050315Z.csv`
- artifact_parse_smoke_json: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_artifact_parse_smoke_20260523T050315Z.json`
