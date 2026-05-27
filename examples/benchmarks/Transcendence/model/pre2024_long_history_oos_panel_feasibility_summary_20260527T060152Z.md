# Pre-2024 Long-History OOS Panel Feasibility

- task_id: `Q-LONG-HISTORY-OOS-PANEL-FEASIBILITY`
- verdict: `NO_GO`
- gate_pass: `False`
- panel_generation_feasible_now: `False`
- panel_generated: `False`
- no_2024_plus_data_loaded_or_evaluated: `True`

## Available Entrypoints

| entrypoint | action | feasible now | key blockers |
|---|---|---:|---|
| `long_history_model_retrain` | `NO_RUN` | `False` | Hard-coded TEST_START=2024-01-01; candidate_pred is generated only for the 2024-2026 test split.; Hard-coded TEST_END=2026-04-30; _build_long_history_panel reads feature bins through 2026-04-30.; Entrypoint trains a LightGBMRegressor before prediction; no load-pretrained or predict-only mode is present.; Training uses n_jobs=8 and defaults to n_estimators=800 / num_leaves=127, which is not a bounded smoke job.; Entrypoint runs validation and 2024-2026 test backtests after fitting. |
| `long_history_second_order_ensemble` | `NO_RUN` | `False` | Uses fixed published factor_meta and long_history candidate prediction files, not a generator for new 2020-2021 OOS panels.; Loads full prediction pickle artifacts before slicing, which is not acceptable for a strict no-2024+ feasibility run.; Runs same-window portfolio backtests after creating the ensemble signal. |

## Required Changes

- Add CLI-controlled pre-2024 train/apply windows with hard max-date guards.
- Add panel-export-only smoke mode that skips all portfolio backtests.
- Add explicit resource bounds: max instruments/dates, <=20 estimators, <=2 jobs, and one fold.
- Avoid full post-2023 pickle loads; use pre-filtered CSV or freshly generated pre-2024-only output.

## Estimates

- this audit runtime/memory: `<5s` / `<100MB`.
- current retrain entrypoint: `NO-GO`, large LightGBM plus backtests and unbounded dense panel memory.
- bounded smoke after changes: target `<=5-10min` and `<=2GB`.

## Outputs

- summary_json: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\model\pre2024_long_history_oos_panel_feasibility_summary_20260527T060152Z.json`
- entrypoints_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\model\pre2024_long_history_oos_panel_feasibility_entrypoints_20260527T060152Z.csv`
- artifacts_csv: `C:\Users\15728\Desktop\Quant-Master-Research\examples\benchmarks\Transcendence\model\pre2024_long_history_oos_panel_feasibility_artifacts_20260527T060152Z.csv`