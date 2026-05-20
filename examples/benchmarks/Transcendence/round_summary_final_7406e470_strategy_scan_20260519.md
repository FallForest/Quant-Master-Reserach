# Round Summary Final: 7406 Strategy-Level SOTA Verification (2026-05-19)

## Scope

- Role: Final Verifier/Artifact Worker
- Constraint: only Transcendence artifacts updated; no business/model code modified
- Subject:
  - Candidate A: `buffered_tk55_hk85_equal_weekly`
  - Candidate B: rank ensemble combined scan best rows

## Independent Verification

1. Candidate A existence and metrics were confirmed in both:
   - `examples/benchmarks/Transcendence/portfolio_innov_ext_i10_summary_7406e470_20260519T122816Z.json`
   - `examples/benchmarks/Transcendence/portfolio_innov_ext_i10_7406e470_20260519T122816Z.csv`
2. Candidate A confirmed values:
   - costed IR: `3.0230019401859436`
   - costed AnnRet: `0.3878544154715252`
   - max drawdown: `-0.047723124283069907`
   - turnover: `0.25471396243602523`
3. Consistency check passed:
   - open_cost: `0.0005`
   - close_cost: `0.0015`
   - benchmark: `SH000300`
   - test period: `2024-01-01` to `2026-04-30`
4. Candidate B confirmed from:
   - `examples/benchmarks/Transcendence/rank_ensemble_scan_combined_summary_20260519T132041Z.json`
   - best IR: `2.843382353055041`
   - best AnnRet: `0.3037207388925102`
   - below Candidate A on both IR and AnnRet.

## Single-Point Re-evaluation (Independent)

- Method: direct one-combo call to `scan_portfolio_innovations._run_one_scan` for `buffered_tk55_hk85_equal_weekly`.
- Confirmed (recomputed) metrics:
  - costed IR: `3.023001940185944`
  - costed AnnRet: `0.38785441547152355`
  - max drawdown: `-0.04772312428306989`
  - turnover: `0.2547139624360253`
- Numerical differences vs summary are only floating-point noise.

## Final Decision

- Set Candidate A as current portfolio SOTA in artifacts.
- Classification: strategy-level offline scan on existing signal (`7406e470`), not a new model retraining result.
- Risk note retained: parameter selection happened on the test scan; walk-forward validation is required to control overfitting risk.
