# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

QuantMaster (`pyquant_master` on PyPI) is an AI-oriented quantitative investment platform forked from Microsoft's Qlib. It covers the full ML pipeline: data processing, feature engineering, model training, backtesting, portfolio optimization, and order execution.

## Common Commands

### Install & Build
```bash
make install          # Compile Cython extensions + pip install -e .
make dev              # Install all optional dependency groups
make prerequisite     # Compile Cython .pyx → .so/.pyd only
```

### Run Tests
```bash
cd tests
python -m pytest . -m "not slow" --durations=0    # Fast tests
python -m pytest . -m "slow" --durations=0          # Slow tests
python -m pytest test_all_pipeline.py               # Single test file
python -m pytest test_all_pipeline.py::TestClass::test_method -v  # Single test
```

Tests require data downloaded first:
```bash
python scripts/get_data.py quant_master_data --name quant_master_data_simple --target_dir ~/.quant_master/quant_master_data/tdx_cn_data --interval 1d --region cn
```

Current unified CN data directory: `~/.quant_master/quant_master_data/tdx_cn_data`
(Windows runtime path: `C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data`).

RL tests (`tests/rl/`) are Linux-only and auto-skipped on other platforms.

### Lint & Format
```bash
make lint             # Run all: black, pylint, flake8, mypy, nbqa
make black            # black . -l 120 --check --diff
make flake8           # flake8 with project ignore list
make mypy             # mypy quant_master (mostly backtest/ only)
```

Line length: **120**. Flake8 ignores: `E501,F541,E266,E402,W503,E731,E203`.

### Run a Workflow
```bash
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
# or
python quant_master/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

### Data Collection (Yahoo example)
```bash
python scripts/data_collector/yahoo/collector.py update_data_to_bin --quant_master_data_1d_dir ~/.quant_master/quant_master_data/tdx_cn_data --end_date 2025-01-01
```

## Architecture

### Global Singletons
- **`C`** (`quant_master.config`): Global configuration — data URIs, Redis, MLflow, region settings. Env vars use `QUANT_MASTER_` prefix.
- **`D`** (`quant_master.data`): Data accessor wrapping `LocalProvider` or `ClientProvider`. Primary API: `D.features(instruments, fields)`.
- **`R`** (`quant_master.workflow`): Experiment recorder wrapper around `QuantMasterRecorder` (MLflow-backed).
- **`H`** (`quant_master.data.cache`): Cache manager.

### Initialization
```python
import quant_master
quant_master.init(provider_uri="~/.quant_master/quant_master_data/tdx_cn_data", region="cn")
```

### Package Layout

**`quant_master/data/`** — Core data layer. Binary storage format with `calendars/`, `instruments/`, `features/` directories. `ops.py` provides expression operators for feature engineering. `dataset/` wraps data into ML-ready splits. `_libs/` has Cython rolling/expanding window ops.

**`quant_master/backtest/`** — Backtesting engine: `exchange.py` (order matching), `executor.py` (execution), `account.py`/`position.py` (state), `backtest.py` (main loop). Exports `backtest()`, `collect_data()`, `Order`.

**`quant_master/model/`** — ML abstractions: `Model` (fit/predict), `ModelFT` (fine-tunable), `trainer.py` (orchestration via `task_train`).

**`quant_master/contrib/`** — Concrete implementations extending core abstractions:
- `model/` — 50+ models (LightGBM, XGBoost, LSTM, GRU, Transformer, GAT, TCN, TFT, TabNet, ensemble variants)
- `data/` — `Alpha158`/`Alpha360` handlers, high-freq handlers, processors, loaders
- `strategy/` — Signal strategies, rule strategies, cost-aware strategies, optimizer
- `report/` — Analysis and reporting utilities

**`quant_master/workflow/`** — Experiment management: `Experiment`, `Recorder`, `ExpManager` (MLflow), `record_temp.py` (signal/analysis recording).

**`quant_master/rl/`** — Reinforcement learning framework: interpreters, rewards, simulators, order execution strategies.

**`quant_master/utils/`** — `init_instance_by_config()` is the universal factory function used throughout to instantiate classes from YAML config dicts. `Wrapper`/`register_wrapper` pattern for global singletons.

**`scripts/`** — Data collection pipeline:
- `data_collector/yahoo/` — Yahoo Finance collector (download → normalize → dump)
- `data_collector/base.py` — Abstract base for all collectors
- `data_collector/utils.py` — Shared HTTP client (`yahoo_fetch()`), symbol conversion, calendar fetching
- `dump_bin.py` — CSV to binary format conversion (`DumpDataAll`, `DumpDataUpdate`)
- `get_data.py` — CLI for downloading pre-built datasets

### YAML Workflow Config Structure

Workflow configs (in `examples/benchmarks/`) define:
- `task.model` — Model class and kwargs
- `task.dataset` — Dataset handler and data loader config
- `task.record` — What to record (signals, predictions)
- `strategy` — Backtest strategy config
- `backtest` — Exchange, executor, account settings

Configs support Jinja2 templating from env vars and `BASE_CONFIG_PATH` for inheritance.

## Commit Convention

Uses conventional commits (`.commitlintrc.js`). Types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`. Max header: 100 chars.
