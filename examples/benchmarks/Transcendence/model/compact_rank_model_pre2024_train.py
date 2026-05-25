#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import quant_master
import pre2024_train_new_model_lockstep as base


RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    yz = np.nan_to_num(y - np.nanmean(y), nan=0.0, posinf=0.0, neginf=0.0)
    xtx = xz.T @ xz
    coef = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=np.float64) * float(alpha), xz.T @ yz)
    return coef.astype(np.float64), mu.astype(np.float64), sd.astype(np.float64)


def _predict_ridge(x: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return xz @ coef


def _daily_rank_ic_series(pred: pd.Series, label: pd.Series) -> pd.Series:
    panel = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    vals: List[Tuple[pd.Timestamp, float]] = []
    for dt, g in panel.groupby(level=0, sort=False):
        if len(g) < 20:
            continue
        corr = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(corr):
            vals.append((pd.Timestamp(dt), float(corr)))
    if not vals:
        return pd.Series(dtype=float)
    return pd.Series({dt: val for dt, val in vals}, dtype=float).sort_index()


def _mean_and_ir(s: pd.Series) -> Tuple[float, float]:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return float("nan"), float("nan")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    return mean, float(mean / (std + 1e-12) * np.sqrt(252.0))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact strict pre-2024 ridge rank model search.")
    p.add_argument("--provider-uri", default=".qmData/cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default=str(
            THIS_DIR / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
        ),
    )
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--alpha-grid", default="0.1,1,10,100,1000,10000")
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    p.add_argument("--preselect", type=int, default=6)
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="compact_rank_model_pre2024_train")
    p.add_argument("--smoke", action="store_true", help="Use a two-alpha, one-combo run for plumbing validation.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    quant_master.init(provider_uri=str(provider_uri), region="cn")

    workflow_cfg = base._load_config(Path(args.workflow_config).expanduser().resolve())
    port_cfg = base._extract_port_config(workflow_cfg)
    benchmark = str(workflow_cfg.get("benchmark", "SH000300"))

    panel_raw, coverage_df = base._build_panel(provider_uri, str(args.market), RAW_START, TEST_END, base.BASE_FIELDS)
    dataset, feature_cols = base._build_features_and_targets(panel_raw)
    dataset = dataset.dropna(subset=["label_raw", "label_rank", "label_volnorm_rank"])
    day_counts = dataset.groupby(level=0)["label_raw"].count()
    good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
    dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    train_df = dataset.loc[_mask(dt_idx, TRAIN_START, TRAIN_END)]
    valid_df = dataset.loc[_mask(dt_idx, VALID_START, VALID_END)]
    test_df = dataset.loc[_mask(dt_idx, TEST_START, TEST_END)]
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(f"empty split train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

    alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
    topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
    ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
    if args.smoke:
        alpha_grid = alpha_grid[:2]
        topk_grid = topk_grid[:1]
        ndrop_grid = ndrop_grid[:1]

    feature_modes = {
        "rank_only": [c for c in feature_cols if c.endswith("__rank")],
        "rank_and_z": list(feature_cols),
    }
    target_modes = ["label_rank", "label_volnorm_rank"]
    candidate_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}

    for feature_mode, cols in feature_modes.items():
        x_train = train_df[cols].astype(np.float64).values
        x_valid = valid_df[cols].astype(np.float64).values
        x_test = test_df[cols].astype(np.float64).values
        for target_mode in target_modes:
            y_train = train_df[target_mode].astype(np.float64).values
            for alpha in alpha_grid:
                candidate_id = f"ridge_{feature_mode}_{target_mode}_a{alpha:g}"
                fit_t0 = time.perf_counter()
                coef, mu, sd = _fit_ridge(x_train, y_train, alpha)
                fit_sec = time.perf_counter() - fit_t0
                pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
                pred_valid = _cs_z(pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
                pred_test = _cs_z(pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"))
                train_ic_s = _daily_rank_ic_series(pred_train, train_df["label_raw"])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df["label_raw"])
                train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                predictions[candidate_id] = {"train": pred_train, "valid": pred_valid, "test": pred_test}
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_family": "closed_form_ridge_rank",
                        "feature_mode": feature_mode,
                        "target_mode": target_mode,
                        "alpha": alpha,
                        "feature_count": len(cols),
                        "fit_sec": fit_sec,
                        "train_rank_ic": train_ic,
                        "train_rank_ic_ir": train_ic_ir,
                        "valid_rank_ic": valid_ic,
                        "valid_rank_ic_ir": valid_ic_ir,
                    }
                )

    preselected = sorted(
        candidate_rows,
        key=lambda r: (
            _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
            _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
        ),
        reverse=True,
    )[: max(1, int(args.preselect))]

    combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk]
    if not combos:
        combos = [(40, 2)]

    exchange_cache: Dict[Tuple[str, str, int, int], Any] = {}
    valid_bt_rows: List[Dict[str, Any]] = []
    for cand in preselected:
        sig = predictions[str(cand["candidate_id"])]["valid"].rename("score").to_frame("score")
        for topk, n_drop in combos:
            metric = base._run_bt(
                sig,
                "valid_2023",
                str(cand["candidate_id"]),
                VALID_START,
                VALID_END,
                int(topk),
                int(n_drop),
                port_cfg,
                benchmark,
                float(args.open_cost),
                float(args.close_cost),
                exchange_cache,
            )
            valid_bt_rows.append(
                {
                    "split": metric.split,
                    "candidate_id": metric.candidate_id,
                    "topk": metric.topk,
                    "n_drop": metric.n_drop,
                    "costed_ir": metric.ir,
                    "costed_annret": metric.annret,
                    "max_drawdown": metric.max_drawdown,
                    "turnover": metric.turnover,
                    "elapsed_sec": metric.elapsed_sec,
                    "error": metric.error,
                }
            )

    ok_valid = [
        r
        for r in valid_bt_rows
        if not r["error"] and np.isfinite(_safe_float(r["costed_ir"])) and np.isfinite(_safe_float(r["costed_annret"]))
    ]
    selected = sorted(ok_valid, key=lambda r: (_safe_float(r["costed_ir"]), _safe_float(r["costed_annret"])), reverse=True)[0]
    selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])

    test_metric = base._run_bt(
        predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score"),
        "test_2024_2026",
        str(selected["candidate_id"]),
        TEST_START,
        TEST_END,
        int(selected["topk"]),
        int(selected["n_drop"]),
        port_cfg,
        benchmark,
        float(args.open_cost),
        float(args.close_cost),
        exchange_cache,
    )
    test_ic_s = _daily_rank_ic_series(predictions[str(selected["candidate_id"])]["test"], test_df["label_raw"])
    test_rank_ic, test_rank_ic_ir = _mean_and_ir(test_ic_s)
    hard_gate_pass = bool(
        np.isfinite(_safe_float(selected["costed_ir"]))
        and np.isfinite(_safe_float(selected["costed_annret"]))
        and _safe_float(selected["costed_ir"]) > HARD_GATE_IR
        and _safe_float(selected["costed_annret"]) > HARD_GATE_ANNRET
    )

    coverage_csv = THIS_DIR / f"{args.output_prefix}_coverage_{stamp}.csv"
    candidates_csv = THIS_DIR / f"{args.output_prefix}_candidates_{stamp}.csv"
    selection_csv = THIS_DIR / f"{args.output_prefix}_validation_selection_{stamp}.csv"
    split_metrics_csv = THIS_DIR / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    pred_pkl = THIS_DIR / f"{args.output_prefix}_candidate_pred_{stamp}.pkl"
    pred_csv = THIS_DIR / f"{args.output_prefix}_candidate_pred_{stamp}.csv"
    summary_json = THIS_DIR / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = THIS_DIR / f"{args.output_prefix}_summary_{stamp}.md"
    smoke_json = THIS_DIR / f"{args.output_prefix}_artifact_parse_smoke_{stamp}.json"

    coverage_df.to_csv(coverage_csv, index=False)
    _write_csv(candidates_csv, candidate_rows)
    _write_csv(selection_csv, valid_bt_rows)

    split_rows = [
        {
            "split": "train_2020_2022",
            "candidate_id": selected["candidate_id"],
            "rank_ic": selected_candidate["train_rank_ic"],
            "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
            "costed_ir": "",
            "costed_annret": "",
            "max_drawdown": "",
            "turnover": "",
            "topk": "",
            "n_drop": "",
            "error": "",
        },
        {
            "split": "valid_2023",
            "candidate_id": selected["candidate_id"],
            "rank_ic": selected_candidate["valid_rank_ic"],
            "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
            "costed_ir": selected["costed_ir"],
            "costed_annret": selected["costed_annret"],
            "max_drawdown": selected["max_drawdown"],
            "turnover": selected["turnover"],
            "topk": selected["topk"],
            "n_drop": selected["n_drop"],
            "error": selected["error"],
        },
        {
            "split": "test_2024_2026",
            "candidate_id": selected["candidate_id"],
            "rank_ic": test_rank_ic,
            "rank_ic_ir": test_rank_ic_ir,
            "costed_ir": test_metric.ir,
            "costed_annret": test_metric.annret,
            "max_drawdown": test_metric.max_drawdown,
            "turnover": test_metric.turnover,
            "topk": test_metric.topk,
            "n_drop": test_metric.n_drop,
            "error": test_metric.error,
        },
    ]
    _write_csv(split_metrics_csv, split_rows)

    pred_test = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
    with pred_pkl.open("wb") as f:
        pickle.dump(pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    pred_test.reset_index().to_csv(pred_csv, index=False)

    summary = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "strict compact rank model; train 2020-2022, select on 2023 only, evaluate frozen selection on 2024-2026",
        "provider_uri": str(provider_uri),
        "market": str(args.market),
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "training_protocol": {
            "raw_start": RAW_START,
            "model_family": "closed_form_ridge_rank",
            "feature_source": "local .qmData OHLCV/factor bins; expanded past-only rank/z factors",
            "feature_modes": list(feature_modes.keys()),
            "target_modes": target_modes,
            "alpha_grid": alpha_grid,
            "selection_rule": "max 2023 real net-cost IR, tie by 2023 AnnRet; no test metric used for selection",
            "selected_pre_backtest_by": "2023 daily rank IC IR shortlist only",
            "preselected_candidates": [r["candidate_id"] for r in preselected],
            "topk_grid": topk_grid,
            "ndrop_grid": ndrop_grid,
        },
        "splits": {
            "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": int(len(train_df))},
            "valid": {"start": VALID_START, "end": VALID_END, "rows": int(len(valid_df))},
            "test": {"start": TEST_START, "end": TEST_END, "rows": int(len(test_df))},
        },
        "selected_model": {
            "candidate_id": selected["candidate_id"],
            "feature_mode": selected_candidate["feature_mode"],
            "target_mode": selected_candidate["target_mode"],
            "alpha": selected_candidate["alpha"],
            "feature_count": selected_candidate["feature_count"],
            "topk": selected["topk"],
            "n_drop": selected["n_drop"],
        },
        "metrics": {
            "train_2020_2022": {
                "rank_ic": selected_candidate["train_rank_ic"],
                "rank_ic_ir": selected_candidate["train_rank_ic_ir"],
            },
            "valid_2023": {
                "rank_ic": selected_candidate["valid_rank_ic"],
                "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                "costed_ir": selected["costed_ir"],
                "costed_annret": selected["costed_annret"],
                "max_drawdown": selected["max_drawdown"],
                "turnover": selected["turnover"],
                "error": selected["error"],
            },
            "test_2024_2026": {
                "rank_ic": test_rank_ic,
                "rank_ic_ir": test_rank_ic_ir,
                "costed_ir": test_metric.ir,
                "costed_annret": test_metric.annret,
                "max_drawdown": test_metric.max_drawdown,
                "turnover": test_metric.turnover,
                "error": test_metric.error,
            },
        },
        "hard_gate": {
            "rule": {"scope": "valid_2023_non_test_selection", "ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "passed": hard_gate_pass,
        },
        "runtime_sec_total": float(time.perf_counter() - t0_all),
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "coverage_csv": str(coverage_csv),
            "candidates_csv": str(candidates_csv),
            "validation_selection_csv": str(selection_csv),
            "split_metrics_csv": str(split_metrics_csv),
            "candidate_pred_pkl": str(pred_pkl),
            "candidate_pred_csv": str(pred_csv),
            "artifact_parse_smoke_json": str(smoke_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                f"# Compact Rank Model Pre-2024 Train ({stamp})",
                "",
                f"- hard_gate_pass: `{hard_gate_pass}`",
                f"- selected: `{selected['candidate_id']}` topk/n_drop `{selected['topk']}/{selected['n_drop']}`",
                f"- valid_2023 IR/AnnRet: `{_safe_float(selected['costed_ir']):.6f}` / `{_safe_float(selected['costed_annret']):.6f}`",
                f"- test_2024_2026 IR/AnnRet: `{_safe_float(test_metric.ir):.6f}` / `{_safe_float(test_metric.annret):.6f}`",
                f"- costs: open `{args.open_cost}` close `{args.close_cost}`",
                f"- protocol: train `{TRAIN_START}..{TRAIN_END}`, select `{VALID_START}..{VALID_END}`, test once `{TEST_START}..{TEST_END}`",
                f"- artifacts: `{summary_json}`",
            ]
        ),
        encoding="utf-8",
    )
    smoke = {
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "coverage_csv_rows": int(len(pd.read_csv(coverage_csv))) if coverage_csv.exists() else 0,
        "candidates_csv_rows": int(len(pd.read_csv(candidates_csv))) if candidates_csv.exists() else 0,
        "validation_selection_csv_rows": int(len(pd.read_csv(selection_csv))) if selection_csv.exists() else 0,
        "split_metrics_csv_rows": int(len(pd.read_csv(split_metrics_csv))) if split_metrics_csv.exists() else 0,
        "candidate_pred_rows": int(len(_load_pickle(pred_pkl))) if pred_pkl.exists() else 0,
        "hard_gate_passed": hard_gate_pass,
    }
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
