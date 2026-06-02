# Causal Regime Residual Model 20260523T051041Z

Verdict: **NO_GO**

## Protocol
- Strict quarterly past-only selection.
- Daily stress gate uses previous-day diagnostics only.
- No runtime sell wrapper or execution-layer patch is applied in this script.

## Key Metrics
- Eval window: 2025-01-01..2026-04-28
- Candidate backtest: {"ok": false, "metrics": null, "error": {"type": "Skipped", "message": "--skip-backtest"}}
- Factor-meta same-window backtest: {"ok": false, "metrics": null, "error": {"type": "Skipped", "message": "--skip-backtest"}}
- Candidate proxy objective: 4.582213
- Factor-meta proxy objective: 4.647528
- Proxy objective delta: -0.065316

## Comparison
- factor_augmented_meta published full: {"annret": 0.3386261002570873, "ir": 2.8208783790760705, "max_drawdown": -0.04089447967616651, "turnover": 0.2522513294744337, "elapsed_sec": 7.671771799999988}
- long_history_second_order verdict: NO_GO

## Artifacts
- summary_json: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_summary_20260523T051041Z.json`
- summary_md: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_summary_20260523T051041Z.md`
- candidate_pred_pkl: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_candidate_pred_20260523T051041Z.pkl`
- candidate_pred_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_candidate_pred_20260523T051041Z.csv`
- periods_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_periods_20260523T051041Z.csv`
- selector_diag_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_selector_diag_20260523T051041Z.csv`
- slices_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_slices_20260523T051041Z.csv`
- artifact_parse_smoke_json: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\causal_regime_residual_tiny_proxy_artifact_parse_smoke_20260523T051041Z.json`
