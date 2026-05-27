import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks.Transcendence._bootstrap import ensure_repo_and_benchmark_on_path

ensure_repo_and_benchmark_on_path(__file__)

from examples.benchmarks.Transcendence.support import regime_switch_stability_eval as _impl
from examples.benchmarks.Transcendence.support.regime_switch_stability_eval import *  # noqa: F401,F403


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _finite_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        raw = obj.iloc[:, 0] if obj.shape[1] else pd.Series(dtype=float)
    elif isinstance(obj, pd.Series):
        raw = obj
    else:
        raw = pd.Series(obj)
    numeric = pd.to_numeric(raw, errors="coerce").dropna()
    return numeric[numeric.map(lambda x: math.isfinite(float(x)))]


def _series_mean_count(obj: Any) -> Tuple[Optional[float], int]:
    s = _finite_series(obj)
    if s.empty:
        return None, 0
    return float(s.mean()), int(len(s))


def _first_numeric_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.empty or obj.shape[1] == 0:
            return pd.Series(dtype=float)
        return pd.to_numeric(obj.iloc[:, 0], errors="coerce")
    if isinstance(obj, pd.Series):
        return pd.to_numeric(obj, errors="coerce")
    return pd.to_numeric(pd.Series(obj), errors="coerce")


def _daily_signal_metrics_from_pred_label(pred: Any, label: Any) -> Dict[str, Any]:
    pred_s = _first_numeric_series(pred).rename("pred")
    label_s = _first_numeric_series(label).rename("label")
    panel = pd.concat([pred_s, label_s], axis=1).dropna()
    if panel.empty or not isinstance(panel.index, pd.MultiIndex):
        return {
            "ic": None,
            "rank_ic": None,
            "source": "artifacts.pred.pkl+label.pkl",
            "reason": "empty or non-panel pred/label",
        }

    date_level: Any = "datetime" if "datetime" in panel.index.names else 0
    ic_vals = []
    rank_ic_vals = []
    for _, group in panel.groupby(level=date_level, sort=False):
        if len(group) < 2:
            continue
        ic = group["pred"].corr(group["label"])
        rank_ic = group["pred"].corr(group["label"], method="spearman")
        if pd.notna(ic) and math.isfinite(float(ic)):
            ic_vals.append(float(ic))
        if pd.notna(rank_ic) and math.isfinite(float(rank_ic)):
            rank_ic_vals.append(float(rank_ic))

    result: Dict[str, Any] = {
        "source": "artifacts.pred.pkl+label.pkl",
        "method": "daily Pearson IC and daily Spearman RankIC averaged over finite days",
        "ic_days": len(ic_vals),
        "rank_ic_days": len(rank_ic_vals),
    }
    result["ic"] = float(pd.Series(ic_vals, dtype=float).mean()) if ic_vals else None
    result["rank_ic"] = float(pd.Series(rank_ic_vals, dtype=float).mean()) if rank_ic_vals else None
    if result["ic"] is None or result["rank_ic"] is None:
        result["reason"] = "insufficient finite daily signal correlations"
    return result


def _baseline_signal_metrics_from_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
    sig_dir = artifacts_dir / "sig_analysis"
    ic_path = sig_dir / "ic.pkl"
    ric_path = sig_dir / "ric.pkl"
    if ic_path.exists() and ric_path.exists():
        try:
            ic, ic_days = _series_mean_count(_load_pickle(ic_path))
            rank_ic, rank_ic_days = _series_mean_count(_load_pickle(ric_path))
        except Exception as exc:  # noqa: BLE001
            artifact_error = f"{type(exc).__name__}: {exc}"
        else:
            return {
                "ic": ic,
                "rank_ic": rank_ic,
                "source": "artifacts.sig_analysis.ic.pkl+ric.pkl",
                "method": "mean of existing SigAnaRecord daily IC/RIC artifacts",
                "ic_days": ic_days,
                "rank_ic_days": rank_ic_days,
                "source_paths": {"ic": str(ic_path), "rank_ic": str(ric_path)},
                **(
                    {"reason": "empty sig_analysis IC/RIC artifacts"}
                    if ic is None or rank_ic is None
                    else {}
                ),
            }
    else:
        artifact_error = "missing sig_analysis/ic.pkl or sig_analysis/ric.pkl"

    pred_path = artifacts_dir / "pred.pkl"
    label_path = artifacts_dir / "label.pkl"
    if pred_path.exists() and label_path.exists():
        try:
            metrics = _daily_signal_metrics_from_pred_label(_load_pickle(pred_path), _load_pickle(label_path))
        except Exception as exc:  # noqa: BLE001
            return {
                "ic": None,
                "rank_ic": None,
                "source": "unavailable",
                "reason": f"{artifact_error}; pred/label fallback failed: {type(exc).__name__}: {exc}",
            }
        metrics["fallback_reason"] = artifact_error
        metrics["source_paths"] = {"pred": str(pred_path), "label": str(label_path)}
        return metrics

    return {
        "ic": None,
        "rank_ic": None,
        "source": "unavailable",
        "reason": f"{artifact_error}; missing pred.pkl or label.pkl",
    }


def _latest_summary_path(output_prefix: str, run_id: str, out_dir: Path) -> Path:
    run_short = run_id[:8]
    matches = sorted(
        out_dir.glob(f"{output_prefix}_summary_{run_short}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"summary not found for prefix={output_prefix!r} run={run_short}")
    return matches[0]


def _postprocess_summary_signal_metrics(summary_path: Path, tracking_uri: str, run_id: str) -> Dict[str, Any]:
    tracking_dir = _impl._parse_tracking_dir(tracking_uri)
    run_dir = _impl._find_run_dir(tracking_dir, run_id)
    metrics = _baseline_signal_metrics_from_artifacts(run_dir / "artifacts")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["baseline_signal_metrics"] = metrics
    summary["ic"] = metrics.get("ic")
    summary["rank_ic"] = metrics.get("rank_ic")
    summary["signal_metrics_source"] = metrics.get("source")
    if metrics.get("ic") is None:
        summary["ic_missing_reason"] = metrics.get("reason", "unable to compute baseline signal IC")
    else:
        summary.pop("ic_missing_reason", None)
    if metrics.get("rank_ic") is None:
        summary["rank_ic_missing_reason"] = metrics.get("reason", "unable to compute baseline signal RankIC")
    else:
        summary.pop("rank_ic_missing_reason", None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _impl.build_arg_parser()
    args = parser.parse_args(argv)

    if argv is None:
        exit_code = _impl.main()
    else:
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0], *argv]
        try:
            exit_code = _impl.main()
        finally:
            sys.argv = old_argv
    if exit_code != 0:
        return int(exit_code)

    out_dir = Path("examples/benchmarks/Transcendence").resolve()
    summary_path = _latest_summary_path(args.output_prefix, args.run_id, out_dir)
    summary = _postprocess_summary_signal_metrics(summary_path, args.tracking_uri, args.run_id)
    metrics = summary["baseline_signal_metrics"]
    print(
        "[signal-metrics] "
        f"summary={summary_path} source={metrics.get('source')} "
        f"ic={metrics.get('ic')} rank_ic={metrics.get('rank_ic')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
