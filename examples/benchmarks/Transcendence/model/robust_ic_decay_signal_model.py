#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_SITE_PACKAGES = REPO_ROOT / ".venv" / "Lib" / "site-packages"
if WORKSPACE_SITE_PACKAGES.exists() and str(WORKSPACE_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(WORKSPACE_SITE_PACKAGES))

# Some local runners have QuantMaster installed with quant_master/_version.py
# but without a complete setuptools_scm package. Keep this script self-contained
# without editing existing repo files.
try:
    from setuptools_scm import get_version as _unused_get_version  # noqa: F401
except Exception:  # noqa: BLE001
    scm_stub = types.ModuleType("setuptools_scm")
    scm_stub.get_version = lambda *args, **kwargs: "0+local"
    sys.modules["setuptools_scm"] = scm_stub

try:
    from simplejson import JSONDecodeError as _unused_json_decode_error  # noqa: F401
except Exception:  # noqa: BLE001
    json_stub = types.ModuleType("simplejson")
    json_stub.JSONDecodeError = json.JSONDecodeError
    json_stub.dumps = json.dumps
    json_stub.loads = json.loads
    json_stub.dump = json.dump
    json_stub.load = json.load
    sys.modules["simplejson"] = json_stub

try:
    from loguru import logger as _unused_loguru_logger  # noqa: F401
except Exception:  # noqa: BLE001
    class _LoguruLoggerStub:
        def debug(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def exception(self, *args, **kwargs):
            return None

        def warn(self, *args, **kwargs):
            return self.warning(*args, **kwargs)

    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = _LoguruLoggerStub()
    sys.modules["loguru"] = loguru_stub

try:
    from filelock import FileLock as _unused_file_lock  # noqa: F401
except Exception:  # noqa: BLE001
    class _FileLockStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def acquire(self, *args, **kwargs):
            return self

        def release(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    filelock_stub = types.ModuleType("filelock")
    filelock_stub.FileLock = _FileLockStub
    filelock_stub.Timeout = TimeoutError
    sys.modules["filelock"] = filelock_stub

# The only supported action here is replaying saved predictions/backtests. If a
# partial torch namespace exists in a sidecar site-packages directory, make the
# optional QuantMaster PyTorch models skip cleanly while keeping scipy's optional
# array checks satisfied.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

if "mlflow.exceptions" not in sys.modules:
    mlflow_mod = types.ModuleType("mlflow")
    mlflow_mod.__version__ = "0.0.0"

    class _MlflowException(Exception):
        def __init__(self, message="", error_code=None):
            super().__init__(message)
            self.error_code = error_code

    class _ErrorCode:
        @staticmethod
        def Name(code):
            return str(code)

    class _ViewType:
        ACTIVE_ONLY = 1
        DELETED_ONLY = 2
        ALL = 3

    class _MlflowClientStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __getattr__(self, name):
            raise _MlflowException(f"MlflowClient stub method not available: {name}")

    mlflow_tracking = types.ModuleType("mlflow.tracking")
    mlflow_tracking.MlflowClient = _MlflowClientStub
    mlflow_entities = types.ModuleType("mlflow.entities")
    mlflow_entities.ViewType = _ViewType
    mlflow_entities.run = types.SimpleNamespace(Run=type("Run", (), {}))
    mlflow_exceptions = types.ModuleType("mlflow.exceptions")
    mlflow_exceptions.MlflowException = _MlflowException
    mlflow_exceptions.RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    mlflow_exceptions.ErrorCode = _ErrorCode
    mlflow_utils = types.ModuleType("mlflow.utils")
    mlflow_utils.validation = types.SimpleNamespace(MAX_PARAM_VAL_LENGTH=1000)
    mlflow_store = types.ModuleType("mlflow.store")
    mlflow_artifact = types.ModuleType("mlflow.store.artifact")
    mlflow_azure = types.ModuleType("mlflow.store.artifact.azure_blob_artifact_repo")
    mlflow_azure.AzureBlobArtifactRepository = type("AzureBlobArtifactRepository", (), {})

    mlflow_mod.tracking = mlflow_tracking
    mlflow_mod.entities = mlflow_entities
    mlflow_mod.exceptions = mlflow_exceptions
    mlflow_mod.utils = mlflow_utils
    mlflow_mod.set_tracking_uri = lambda *args, **kwargs: None
    mlflow_mod.start_run = lambda *args, **kwargs: types.SimpleNamespace(info=types.SimpleNamespace(run_id="stub"))
    mlflow_mod.end_run = lambda *args, **kwargs: None

    sys.modules["mlflow"] = mlflow_mod
    sys.modules["mlflow.tracking"] = mlflow_tracking
    sys.modules["mlflow.entities"] = mlflow_entities
    sys.modules["mlflow.exceptions"] = mlflow_exceptions
    sys.modules["mlflow.utils"] = mlflow_utils
    sys.modules["mlflow.store"] = mlflow_store
    sys.modules["mlflow.store.artifact"] = mlflow_artifact
    sys.modules["mlflow.store.artifact.azure_blob_artifact_repo"] = mlflow_azure

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27

BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
GRU_1A085_RUN_ID = "1a085ff9b5a34f408a44ad74055fc5da"
GRU_773_RUN_ID = "773bd6d8413b4bb0b388a63a6b5b6a86"
RANK_BC641_RUN_ID = "bc641cef654441d2bf0c7008e6c90458"
GRU_BCBECF55_RUN_ID = "bcbecf55a3924357ba93fc55b1140e99"


@dataclass(frozen=True)
class SignalSpec:
    signal_id: str
    kind: str
    run_ids: Tuple[str, ...]
    weights: Tuple[float, ...]
    note: str


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {"error_type": type(exc).__name__, "error_message": str(exc), "traceback_tail": tb}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


class _PicklePlaceholder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return _PicklePlaceholder(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            __import__(module)
            mod = sys.modules[module]
            return getattr(mod, name)
        except Exception:  # noqa: BLE001
            return type(name, (_PicklePlaceholder,), {"__module__": module})


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return conv.yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        with path.open("rb") as f:
            obj = _ConfigUnpickler(f).load()
        if not isinstance(obj, dict):
            raise TypeError(f"config artifact is not a dict: {type(obj)}")
        return obj


def _as_label_series(label_obj: Any) -> pd.Series:
    if isinstance(label_obj, pd.Series):
        return label_obj.astype(float)
    if isinstance(label_obj, pd.DataFrame):
        if "label" in label_obj.columns:
            return label_obj["label"].astype(float)
        return label_obj.iloc[:, 0].astype(float)
    raise TypeError(f"unsupported label type: {type(label_obj)}")


def _normalize_mi_dt_inst(series: pd.Series) -> pd.Series:
    idx = series.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise TypeError("expected MultiIndex(datetime, instrument)")
    dt0 = pd.to_datetime(pd.Index(idx.get_level_values(0)[:32]), errors="coerce")
    dt1 = pd.to_datetime(pd.Index(idx.get_level_values(1)[:32]), errors="coerce")
    out = series
    if dt0.notna().mean() < dt1.notna().mean():
        out = out.swaplevel(0, 1)
    out = out.sort_index()
    out.index = out.index.set_names(["datetime", "instrument"] + list(out.index.names[2:]))
    return out.astype(float)


def _as_score_df(pred_obj: Any) -> pd.DataFrame:
    return conv._as_score_df(pred_obj).sort_index()


def _date_values(obj: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    idx = obj.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    return pd.to_datetime(idx)


def _slice_obj(obj: pd.Series | pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    dates = _date_values(obj)
    return obj.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _coverage(obj: pd.Series | pd.DataFrame, start: pd.Timestamp = TEST_START, end: pd.Timestamp = TEST_END) -> Dict[str, Any]:
    dates = _date_values(obj)
    all_days = pd.Index(dates.normalize().unique()).sort_values()
    mask = (dates >= start) & (dates <= end)
    test_days = pd.Index(dates[mask].normalize().unique()).sort_values()
    return {
        "rows": int(len(obj)),
        "days": int(len(all_days)),
        "start": str(pd.Timestamp(all_days.min()).date()) if len(all_days) else None,
        "end": str(pd.Timestamp(all_days.max()).date()) if len(all_days) else None,
        "test_rows": int(mask.sum()),
        "test_days": int(len(test_days)),
        "test_start": str(pd.Timestamp(test_days.min()).date()) if len(test_days) else None,
        "test_end": str(pd.Timestamp(test_days.max()).date()) if len(test_days) else None,
    }


def _rank_pct(score: pd.Series) -> pd.Series:
    return score.groupby(level=0).rank(method="average", pct=True)


def _center_rank(score: pd.Series) -> pd.Series:
    return 2.0 * _rank_pct(score.astype(float)) - 1.0


def _candidate_specs() -> List[SignalSpec]:
    return [
        SignalSpec("base40", "single", (BASE_RUN_ID,), (1.0,), "base SOTA pred.pkl; fixed 2024H1 anchor"),
        SignalSpec("gru1a085", "single", (GRU_1A085_RUN_ID,), (1.0,), "GRU component run"),
        SignalSpec("gru773", "single", (GRU_773_RUN_ID,), (1.0,), "GRU/rank component run"),
        SignalSpec("gru_bcbecf55", "single", (GRU_BCBECF55_RUN_ID,), (1.0,), "GRU conversion candidate run"),
        SignalSpec(
            "rank_gru45",
            "rank_ensemble",
            (BASE_RUN_ID, GRU_1A085_RUN_ID, GRU_773_RUN_ID),
            (0.40, 0.20, 0.40),
            "predeclared GRU45 rank ensemble",
        ),
        SignalSpec(
            "rank50",
            "rank_ensemble",
            (BASE_RUN_ID, GRU_773_RUN_ID, RANK_BC641_RUN_ID),
            (0.60, 0.20, 0.20),
            "predeclared rank50 ensemble",
        ),
    ]


def _load_run_signal(tracking_dir: Path, run_id: str) -> pd.DataFrame:
    run_dir = conv._find_run_dir(tracking_dir, run_id)
    return _as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))


def _build_rank_ensemble(run_scores: Mapping[str, pd.DataFrame], spec: SignalSpec) -> pd.DataFrame:
    cols = []
    for run_id in spec.run_ids:
        ranked = _rank_pct(run_scores[run_id]["score"].astype(float))
        ranked.name = run_id
        cols.append(ranked)
    panel = pd.concat(cols, axis=1)
    weights = pd.Series(spec.weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(weights, axis=1).sum(axis=1)
    score = panel.mul(weights, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _build_candidate_signals(tracking_dir: Path, specs: Sequence[SignalSpec]) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    run_ids = sorted({rid for spec in specs for rid in spec.run_ids})
    run_scores = {run_id: _load_run_signal(tracking_dir, run_id) for run_id in run_ids}
    signals: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        if spec.kind == "single":
            signal = run_scores[spec.run_ids[0]][["score"]].copy()
        elif spec.kind == "rank_ensemble":
            signal = _build_rank_ensemble(run_scores, spec)
        else:
            raise ValueError(f"unsupported signal kind: {spec.kind}")
        signals[spec.signal_id] = signal.sort_index()
        rows.append(
            {
                "signal_id": spec.signal_id,
                "kind": spec.kind,
                "run_ids": "|".join(spec.run_ids),
                "weights": "|".join(str(x) for x in spec.weights),
                "note": spec.note,
                **{f"coverage_{k}": v for k, v in _coverage(signal).items()},
            }
        )
    return signals, rows


def _daily_ic(signal: pd.Series, label: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sig = _slice_obj(_center_rank(signal), start, end)
    lab = _slice_obj(_center_rank(label), start, end)
    aligned = pd.concat({"signal": sig, "label": lab}, axis=1).dropna()
    vals: Dict[pd.Timestamp, float] = {}
    if aligned.empty:
        return pd.Series(dtype=float)
    for dt, grp in aligned.groupby(level=0):
        if len(grp) < 30:
            continue
        corr = grp["signal"].corr(grp["label"], method="spearman")
        if pd.notna(corr) and np.isfinite(float(corr)):
            vals[pd.Timestamp(dt)] = float(corr)
    return pd.Series(vals, dtype=float).sort_index()


def _ic_feature_row(signal_id: str, ic: pd.Series, train_tag: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "train_tag": train_tag,
        "signal_id": signal_id,
        "train_start": str(start.date()),
        "train_end": str(end.date()),
        "ic_days": int(len(ic)),
        "ic_mean": float("nan"),
        "ic_ir": float("nan"),
        "ic_early_mean": float("nan"),
        "ic_late_mean": float("nan"),
        "ic_recent_ewm": float("nan"),
        "ic_decay": float("nan"),
        "raw_reliability": float("nan"),
    }
    if ic.empty:
        return row
    arr = ic.astype(float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    ic_ir = float(np.sqrt(252.0) * arr.mean() / std) if np.isfinite(std) and std > 1e-12 else float("nan")
    cut = max(1, len(arr) // 2)
    early = arr.iloc[:cut]
    late = arr.iloc[cut:] if cut < len(arr) else arr
    recent = float(arr.ewm(halflife=max(5, min(42, len(arr) // 2)), adjust=False).mean().iloc[-1])
    early_mean = float(early.mean())
    late_mean = float(late.mean())
    decay = float(late_mean - early_mean)
    raw = recent + 0.25 * float(arr.mean()) + 0.015 * (ic_ir if np.isfinite(ic_ir) else 0.0) - 0.50 * max(0.0, -decay)
    row.update(
        {
            "ic_mean": float(arr.mean()),
            "ic_ir": ic_ir,
            "ic_early_mean": early_mean,
            "ic_late_mean": late_mean,
            "ic_recent_ewm": recent,
            "ic_decay": decay,
            "raw_reliability": float(raw),
        }
    )
    return row


def _weights_from_prior_ic(
    *,
    signals: Mapping[str, pd.DataFrame],
    label: pd.Series,
    train_tag: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    apply_start: pd.Timestamp,
    label_embargo_days: int,
    min_ic_days: int,
    max_weight: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], str]:
    rows = []
    raw_scores: Dict[str, float] = {}
    effective_train_end = min(pd.Timestamp(train_end), pd.Timestamp(apply_start) - pd.Timedelta(days=int(label_embargo_days)))
    if effective_train_end < pd.Timestamp(train_start):
        return {"base40": 1.0}, rows, "fallback_base40_empty_after_label_embargo"
    for signal_id, signal_df in signals.items():
        ic = _daily_ic(signal_df["score"].astype(float), label, train_start, effective_train_end)
        row = _ic_feature_row(signal_id, ic, train_tag, train_start, effective_train_end)
        row["nominal_train_end"] = str(pd.Timestamp(train_end).date())
        row["apply_start"] = str(pd.Timestamp(apply_start).date())
        row["label_embargo_days"] = int(label_embargo_days)
        ok = int(row["ic_days"]) >= int(min_ic_days) and np.isfinite(float(row["raw_reliability"]))
        score = max(0.0, float(row["raw_reliability"])) if ok else 0.0
        row["eligible"] = bool(ok)
        row["positive_reliability"] = float(score)
        rows.append(row)
        raw_scores[signal_id] = score

    total = float(sum(raw_scores.values()))
    if total <= 1e-12:
        return {"base40": 1.0}, rows, "fallback_base40_no_positive_prior_ic"

    weights = {sid: val / total for sid, val in raw_scores.items() if val > 0.0}
    capped = False
    if max_weight > 0:
        for _ in range(10):
            over = {sid: w for sid, w in weights.items() if w > max_weight}
            if not over:
                break
            capped = True
            fixed = {sid: max_weight for sid in over}
            rem_ids = [sid for sid in weights if sid not in fixed]
            rem_budget = max(0.0, 1.0 - sum(fixed.values()))
            old_weights = weights
            rem_total = sum(old_weights[sid] for sid in rem_ids)
            weights = dict(fixed)
            if rem_ids and rem_total > 0 and rem_budget > 0:
                weights.update({sid: rem_budget * old_weights[sid] / rem_total for sid in rem_ids})

    norm = float(sum(weights.values()))
    if norm <= 1e-12:
        return {"base40": 1.0}, rows, "fallback_base40_after_cap"
    weights = {sid: float(w / norm) for sid, w in weights.items()}
    return weights, rows, "prior_ic_decay_positive_weights_capped" if capped else "prior_ic_decay_positive_weights"


def _blend_signals(signals: Mapping[str, pd.DataFrame], weights: Mapping[str, float]) -> pd.DataFrame:
    cols = []
    for signal_id, weight in weights.items():
        if weight <= 0:
            continue
        ranked = _rank_pct(signals[signal_id]["score"].astype(float))
        ranked.name = signal_id
        cols.append(ranked)
    if not cols:
        raise ValueError("no positive signal weights")
    panel = pd.concat(cols, axis=1)
    w = pd.Series({sid: float(weights[sid]) for sid in panel.columns}, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    score = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _apply_windows() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str, pd.Timestamp | None, pd.Timestamp | None]]:
    return [
        ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"), "fixed_predeclared_base40", None, None),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1_prior_ic", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"), "selected_by_2024H2_prior_ic", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31"), "selected_by_2025H1_prior_ic", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-30"), "selected_by_2025H2_prior_ic", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
    ]


def _strategy_combo(topk: int, n_drop: int) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(topk),
        "n_drop": int(n_drop),
        "hold_topk": int(topk),
        "weight_mode": "equal",
        "score_power": 1.0,
    }


def _run_continuous_backtest(
    *,
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    topk: int,
    n_drop: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = str(start.date())
    backtest_cfg["end_time"] = str(end.date())
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    pred_slice = conv._slice_pred(pred_df, start, end)
    if pred_slice.empty:
        raise ValueError(f"empty selected signal slice: {start.date()}..{end.date()}")

    strategy_obj = conv._build_strategy_object(
        combo=_strategy_combo(topk, n_drop),
        pred_df=pred_slice,
        base_strategy_kwargs=base_strategy_kwargs,
    )
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = conv.get_exchange(
        freq=freq,
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        deal_price=deal_price,
        limit_threshold=limit_threshold,
        open_cost=float(open_cost),
        close_cost=float(close_cost),
        min_cost=min_cost,
    )

    t0 = time.perf_counter()
    portfolio_metric_dict, _ = conv.run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = conv._get_report_for_day_freq(portfolio_metric_dict)
    report = report.loc[(pd.to_datetime(report.index) >= start) & (pd.to_datetime(report.index) <= end)].copy()
    metrics = _metrics_from_report(report)
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    metrics["report"] = report
    return metrics


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(excess, errors="coerce").dropna().astype(float).sort_index()
    if s.empty:
        raise ValueError("no finite excess return observations")
    risk_df = risk_analysis(s, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, Any]:
    required = {"return", "bench", "cost"}
    missing = sorted(required.difference(report.columns))
    if missing:
        raise KeyError(f"report missing columns: {missing}")
    excess = (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")
    finite = excess.replace([np.inf, -np.inf], np.nan).dropna()
    risk = _metrics_from_excess(finite)
    turnover = pd.to_numeric(report.get("turnover", pd.Series(index=report.index, dtype=float)), errors="coerce")
    return {
        "costed_annret": float(risk["annret"]),
        "costed_ir": float(risk["ir"]),
        "max_drawdown": float(risk["max_drawdown"]),
        "turnover": float(turnover.reindex(finite.index).mean()) if not finite.empty else float("nan"),
        "rows": int(len(report)),
        "finite_rows": int(len(finite)),
        "nonfinite_rows": int(len(report) - len(finite)),
        "excess": finite,
    }


def _split_metric_rows(report: pd.DataFrame, selection_by_day: Mapping[pd.Timestamp, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    splits = [
        ("test_full", TEST_START, TEST_END),
        ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]
    idx = pd.to_datetime(report.index)
    for tag, start, end in splits:
        part = report.loc[(idx >= start) & (idx <= end)].copy()
        if part.empty:
            continue
        metrics = _metrics_from_report(part)
        days = [pd.Timestamp(x).normalize() for x in part.index]
        selected_counts: Dict[str, int] = {}
        for day in days:
            key = selection_by_day.get(day, "")
            selected_counts[key] = selected_counts.get(key, 0) + 1
        rows.append(
            {
                "split": tag,
                "start": str(start.date()),
                "end": str(end.date()),
                "days": int(metrics["finite_rows"]),
                "costed_annret": float(metrics["costed_annret"]),
                "costed_ir": float(metrics["costed_ir"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "turnover": float(metrics["turnover"]),
                "selected_counts": json.dumps(selected_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _load_frozen_weights(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for row in data.get("selections", []):
        apply_tag = str(row["apply_tag"])
        weights = json.loads(row["weights_json"]) if isinstance(row.get("weights_json"), str) else row.get("weights", {})
        out[apply_tag] = {str(k): float(v) for k, v in weights.items()}
    if not out:
        raise ValueError(f"no frozen selections found in {path}")
    return out


def _build_selected_signal(
    *,
    signals: Mapping[str, pd.DataFrame],
    label: pd.Series,
    min_ic_days: int,
    max_weight: float,
    label_embargo_days: int,
    verify_selection_json: Path | None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], Dict[pd.Timestamp, str]]:
    selected_parts: List[pd.DataFrame] = []
    selection_rows: List[Dict[str, Any]] = []
    ic_rows: List[Dict[str, Any]] = []
    selection_by_day: Dict[pd.Timestamp, str] = {}
    frozen = _load_frozen_weights(verify_selection_json) if verify_selection_json else {}

    for apply_tag, apply_start, apply_end, rule, train_start, train_end in _apply_windows():
        if verify_selection_json:
            if apply_tag not in frozen:
                raise ValueError(f"frozen selection missing apply_tag={apply_tag}")
            weights = frozen[apply_tag]
            reason = "frozen_verification_selection"
        elif apply_tag == "2024H1":
            weights = {"base40": 1.0}
            reason = "fixed_predeclared_base40"
        else:
            assert train_start is not None and train_end is not None
            weights, rows, reason = _weights_from_prior_ic(
                signals=signals,
                label=label,
                train_tag=rule,
                train_start=train_start,
                train_end=train_end,
                apply_start=apply_start,
                label_embargo_days=label_embargo_days,
                min_ic_days=min_ic_days,
                max_weight=max_weight,
            )
            ic_rows.extend(rows)

        blended = _blend_signals(signals, weights)
        part = _slice_obj(blended, apply_start, apply_end)
        if part.empty:
            raise ValueError(f"empty selected signal for {apply_tag}: {apply_start.date()}..{apply_end.date()}")
        part = part.copy()
        part["apply_tag"] = apply_tag
        selected_parts.append(part[["score"]])
        for day in pd.Index(_date_values(part).normalize().unique()):
            selection_by_day[pd.Timestamp(day)] = apply_tag
        selection_rows.append(
            {
                "apply_tag": apply_tag,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "selection_rule": rule,
                "train_start": str(train_start.date()) if train_start is not None else "",
                "train_end": str(train_end.date()) if train_end is not None else "",
                "train_window_strictly_before_apply": bool(train_end is None or train_end < apply_start),
                "selection_reason": reason,
                "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "nonzero_signal_count": int(sum(1 for v in weights.values() if float(v) > 0)),
                "selected_signal_rows": int(len(part)),
                "selected_signal_days": int(pd.Index(_date_values(part).normalize().unique()).size),
            }
        )

    selected = pd.concat(selected_parts).sort_index()
    duplicated = selected.index.duplicated().sum()
    if duplicated:
        raise ValueError(f"selected signal has duplicated index rows: {duplicated}")
    return selected, selection_rows, ic_rows, selection_by_day


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict causal IC-decay signal combiner for 2024+ Transcendence gate.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--n-drop", type=int, default=2)
    p.add_argument("--min-ic-days", type=int, default=40)
    p.add_argument("--max-weight", type=float, default=0.55)
    p.add_argument("--label-embargo-days", type=int, default=2)
    p.add_argument("--output-prefix", default="robust_ic_decay_signal_model")
    p.add_argument("--verify-selection-json", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    trans_dir = Path(__file__).resolve().parent
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    coverage_csv = trans_dir / f"{args.output_prefix}_coverage_{stamp}.csv"
    ic_csv = trans_dir / f"{args.output_prefix}_ic_decay_{stamp}.csv"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    split_csv = trans_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    report_csv = trans_dir / f"{args.output_prefix}_continuous_report_{stamp}.csv"
    selected_signal_pkl = trans_dir / f"{args.output_prefix}_selected_signal_{stamp}.pkl"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    cfg = _load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(cfg)
    base_port_cfg = conv._extract_port_config(cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    label_path = base_dir / "artifacts" / "label.pkl"
    if not label_path.exists():
        raise FileNotFoundError(f"required label data not found: {label_path}")
    label = _normalize_mi_dt_inst(_as_label_series(_load_pickle(label_path)))
    label_coverage = _coverage(label)
    if int(label_coverage["test_days"]) <= 0:
        raise RuntimeError(f"no label/return data available in test window: {label_coverage}")

    specs = _candidate_specs()
    signals, coverage_rows = _build_candidate_signals(tracking_dir, specs)
    missing_signal_ids = [sid for sid, sig in signals.items() if int(_coverage(sig)["test_days"]) <= 0]
    if missing_signal_ids:
        raise RuntimeError(f"signals missing 2024-2026 coverage: {missing_signal_ids}")
    coverage_rows.insert(
        0,
        {
            "signal_id": "base_label",
            "kind": "label",
            "run_ids": args.base_run_id,
            "weights": "",
            "note": "realized label.pkl used only after each prior window closes",
            **{f"coverage_{k}": v for k, v in label_coverage.items()},
        },
    )

    verify_selection_json = Path(args.verify_selection_json) if args.verify_selection_json else None
    selected_signal, selection_rows, ic_rows, selection_by_day = _build_selected_signal(
        signals=signals,
        label=label,
        min_ic_days=int(args.min_ic_days),
        max_weight=float(args.max_weight),
        label_embargo_days=int(args.label_embargo_days),
        verify_selection_json=verify_selection_json,
    )
    with selected_signal_pkl.open("wb") as f:
        pickle.dump(selected_signal[["score"]], f)

    bt = _run_continuous_backtest(
        pred_df=selected_signal[["score"]],
        base_port_cfg=base_port_cfg,
        base_strategy_kwargs=base_strategy_kwargs,
        open_cost=float(args.open_cost),
        close_cost=float(args.close_cost),
        topk=int(args.topk),
        n_drop=int(args.n_drop),
        start=TEST_START,
        end=TEST_END,
    )
    report = bt["report"]
    report.to_csv(report_csv)
    split_rows = _split_metric_rows(report, selection_by_day)

    strict_non_test_selected = bool(
        all(bool(row["train_window_strictly_before_apply"]) for row in selection_rows)
        and selection_rows[0]["selection_reason"] in {"fixed_predeclared_base40", "frozen_verification_selection"}
        and len(selection_rows) == len(_apply_windows())
    )
    hard_gate_pass = bool(
        strict_non_test_selected
        and float(bt["costed_ir"]) > HARD_GATE_IR
        and float(bt["costed_annret"]) > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(coverage_csv, coverage_rows)
    _write_csv(ic_csv, ic_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(split_csv, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "strict_causal_ic_decay_signal_combiner",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "verification_mode": bool(verify_selection_json),
        "verify_selection_json": str(verify_selection_json) if verify_selection_json else "",
        "strict_non_test_selected": strict_non_test_selected,
        "protocol": {
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "open_cost": float(args.open_cost),
            "close_cost": float(args.close_cost),
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "portfolio_rule": {"family": "topk_dropout", "topk": int(args.topk), "n_drop": int(args.n_drop), "rebalance_mode": "daily"},
            "selection_rule": "2024H1 fixed base40; later windows use only immediately prior window realized label IC",
            "ic_decay_formula": "max(0, ewm_recent_ic + 0.25*mean_ic + 0.015*ic_ir - 0.50*max(0,-late_minus_early_ic))",
            "max_weight": float(args.max_weight),
            "min_ic_days": int(args.min_ic_days),
            "label_embargo_days": int(args.label_embargo_days),
        },
        "label_coverage": label_coverage,
        "candidate_specs": [spec.__dict__ for spec in specs],
        "selections": selection_rows,
        "continuous_full_metrics": {
            "costed_annret": float(bt["costed_annret"]),
            "costed_ir": float(bt["costed_ir"]),
            "max_drawdown": float(bt["max_drawdown"]),
            "turnover": float(bt["turnover"]),
            "rows": int(bt["rows"]),
            "finite_rows": int(bt["finite_rows"]),
            "nonfinite_rows": int(bt["nonfinite_rows"]),
            "elapsed_sec": float(bt["elapsed_sec"]),
        },
        "split_metrics": split_rows,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "ic_decay_csv": str(ic_csv),
            "selections_csv": str(selections_csv),
            "split_metrics_csv": str(split_csv),
            "continuous_report_csv": str(report_csv),
            "selected_signal_pkl": str(selected_signal_pkl),
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Robust IC Decay Signal Model {stamp}",
                "",
                f"- verdict: `{verdict}`",
                f"- hard_gate_pass: `{hard_gate_pass}`",
                f"- strict_non_test_selected: `{strict_non_test_selected}`",
                f"- verification_mode: `{bool(verify_selection_json)}`",
                f"- costed_ir: `{bt['costed_ir']}`",
                f"- costed_annret: `{bt['costed_annret']}`",
                f"- max_drawdown: `{bt['max_drawdown']}`",
                f"- turnover: `{bt['turnover']}`",
                f"- summary_json: `{summary_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "hard_gate_pass": hard_gate_pass,
                "strict_non_test_selected": strict_non_test_selected,
                "costed_ir": float(bt["costed_ir"]),
                "costed_annret": float(bt["costed_annret"]),
                "summary_json": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
