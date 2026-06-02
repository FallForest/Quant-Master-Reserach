# Metrics Extraction Helper

`extract_metrics.py` pulls one run's leaderboard fields:

- `ic`, `rank_ic`
- `costed_annret`, `costed_ir`
- `max_drawdown`, `turnover`
- `runtime_sec`

Priority order:

1. MLflow run metrics (`IC`, `Rank IC`, `1day.excess_return_with_cost.*`)
2. Artifact fallback:
   - `port_analysis_1day.pkl` for `annualized_return/information_ratio/max_drawdown`
   - `report_normal_1day.pkl` for `turnover` mean

## Examples

Latest run in an experiment:

```powershell
python examples\benchmarks\Transcendence\extract_metrics.py --tracking-uri file:./mlruns --experiment-name workflow --format json
```

Specific run:

```powershell
python examples\benchmarks\Transcendence\extract_metrics.py --tracking-uri file:./mlruns --run-id <run_id> --format json
```

Local artifact directory only:

```powershell
python examples\benchmarks\Transcendence\extract_metrics.py --artifact-dir <artifact_dir> --format json
```

CSV append mode:

```powershell
python examples\benchmarks\Transcendence\extract_metrics.py --run-id <run_id> --tracking-uri file:./mlruns --format csv --csv-path examples\benchmarks\Transcendence\leaderboard_rows.csv
```
