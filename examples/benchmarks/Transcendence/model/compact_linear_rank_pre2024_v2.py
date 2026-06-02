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
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import quant_master
from quant_master.config import resolve_provider_uri
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


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


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


def _parse_float_grid(raw: str) -> List[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _parse_int_grid(raw: str) -> List[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strict pre-2024 compact closed-form ridge rank attempt; 2024-2026 is test-only."
    )
    p.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default=str(
            THIS_DIR / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
        ),
    )
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--alpha-grid", default="0.1,1,10,100,1000,10000")
    p.add_argument("--feature-modes", default="rank_only,rank_and_z")
    p.add_argument("--target-modes", default="label_rank,label_volnorm_rank")
    p.add_argument("--topk-grid", default="35,40,45")
    p.add_argument("--ndrop-grid", default="2,3,4")
    p.add_argument("--preselect", type=int, default=6)
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="compact_linear_rank_pre2024_v2")
    p.add_argument("--smoke", action="store_true", help="Limit grids to prove data/backtest path before a wider run.")
    return p


def _make_artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    names = {
        "coverage_csv": "coverage",
        "candidates_csv": "candidates",
        "validation_selection_csv": "validation_selection",
        "split_metrics_csv": "split_metrics",
        "candidate_pred_pkl": "candidate_pred",
        "candidate_pred_csv": "candidate_pred",
        "summary_json": "summary",
        "summary_md": "summary",
        "artifact_parse_smoke_json": "artifact_parse_smoke",
    }
    out: Dict[str, Path] = {}
    for key, suffix in names.items():
        ext = ".pkl" if key.endswith("_pkl") else ".json" if key.endswith("_json") else ".md" if key.endswith("_md") else ".csv"
        out[key] = THIS_DIR / f"{output_prefix}_{suffix}_{stamp}{ext}"
    return out


def _write_failure(paths: Dict[str, Path], summary: Dict[str, Any], exc: BaseException, t0_all: float) -> None:
    summary.update(
        {
            "status": "failed",
            "blocker": f"{type(exc).__name__}: {exc}",
            "runtime_sec_total": float(time.perf_counter() - t0_all),
        }
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary_md"].write_text(
        "\n".join(
            [
                "# Compact Linear Rank Pre-2024 V2",
                "",
                "- status: `failed`",
                f"- blocker: `{summary['blocker']}`",
                f"- artifacts: `{paths['summary_json']}`",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _make_artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))

    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "objective": "strict pre-2024 closed-form ridge rank model; train 2020-2022, select 2023 only, test 2024-2026 once",
        "status": "started",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
    }

    try:
        quant_master.init(provider_uri=str(provider_uri), region="cn")
        workflow_cfg = base._load_config(Path(args.workflow_config).expanduser().resolve())
        port_cfg = base._extract_port_config(workflow_cfg)
        benchmark = str(workflow_cfg.get("benchmark", "SH000300"))

        raw_panel, coverage_df = base._build_panel(provider_uri, str(args.market), RAW_START, TEST_END, base.BASE_FIELDS)
        dataset, feature_cols = base._build_features_and_targets(raw_panel)
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

        alpha_grid = _parse_float_grid(str(args.alpha_grid))
        topk_grid = _parse_int_grid(str(args.topk_grid))
        ndrop_grid = _parse_int_grid(str(args.ndrop_grid))
        requested_feature_modes = [x.strip() for x in str(args.feature_modes).split(",") if x.strip()]
        target_modes = [x.strip() for x in str(args.target_modes).split(",") if x.strip()]
        if args.smoke:
            alpha_grid = alpha_grid[:2]
            topk_grid = topk_grid[:1]
            ndrop_grid = ndrop_grid[:1]
            requested_feature_modes = requested_feature_modes[:1]
            target_modes = target_modes[:1]
            args.preselect = min(int(args.preselect), 2)

        feature_mode_map = {
            "rank_only": [c for c in feature_cols if c.endswith("__rank")],
            "rank_and_z": list(feature_cols),
        }
        selected_feature_modes = {k: feature_mode_map[k] for k in requested_feature_modes if k in feature_mode_map}
        if not selected_feature_modes:
            raise RuntimeError(f"no supported feature modes requested: {requested_feature_modes}")
        unsupported_targets = [t for t in target_modes if t not in train_df.columns]
        if unsupported_targets:
            raise RuntimeError(f"unsupported target modes: {unsupported_targets}")

        candidate_rows: List[Dict[str, Any]] = []
        predictions: Dict[str, Dict[str, pd.Series]] = {}
        for feature_mode, cols in selected_feature_modes.items():
            if not cols:
                raise RuntimeError(f"feature mode {feature_mode} has zero columns")
            x_train = train_df[cols].astype(np.float64).values
            x_valid = valid_df[cols].astype(np.float64).values
            x_test = test_df[cols].astype(np.float64).values
            for target_mode in target_modes:
                y_train = train_df[target_mode].astype(np.float64).values
                for alpha in alpha_grid:
                    candidate_id = f"ridge_{feature_mode}_{target_mode}_a{alpha:g}"
                    fit_t0 = time.perf_counter()
                    coef, mu, sd = _fit_ridge(x_train, y_train, alpha)
                    fit_sec = float(time.perf_counter() - fit_t0)
                    pred_train = _cs_z(
                        pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score")
                    )
                    pred_valid = _cs_z(
                        pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score")
                    )
                    pred_test = _cs_z(
                        pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score")
                    )
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
                            "alpha": float(alpha),
                            "feature_count": int(len(cols)),
                            "fit_sec": fit_sec,
                            "train_rank_ic": train_ic,
                            "train_rank_ic_ir": train_ic_ir,
                            "valid_rank_ic": valid_ic,
                            "valid_rank_ic_ir": valid_ic_ir,
                        }
                    )

        if not candidate_rows:
            raise RuntimeError("no candidates generated")

        preselected = sorted(
            candidate_rows,
            key=lambda r: (
                _safe_float(r["valid_rank_ic_ir"]) if np.isfinite(_safe_float(r["valid_rank_ic_ir"])) else -1e9,
                _safe_float(r["valid_rank_ic"]) if np.isfinite(_safe_float(r["valid_rank_ic"])) else -1e9,
            ),
            reverse=True,
        )[: max(1, int(args.preselect))]
        combos = [(tk, nd) for tk in topk_grid for nd in ndrop_grid if nd < tk] or [(40, 2)]

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
            if not r["error"]
            and np.isfinite(_safe_float(r["costed_ir"]))
            and np.isfinite(_safe_float(r["costed_annret"]))
        ]
        if not ok_valid:
            raise RuntimeError("no valid 2023 backtest combo succeeded; cannot freeze a portfolio")

        selected = sorted(
            ok_valid,
            key=lambda r: (_safe_float(r["costed_ir"]), _safe_float(r["costed_annret"])),
            reverse=True,
        )[0]
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
            not test_metric.error
            and np.isfinite(_safe_float(test_metric.ir))
            and np.isfinite(_safe_float(test_metric.annret))
            and _safe_float(test_metric.ir) > HARD_GATE_IR
            and _safe_float(test_metric.annret) > HARD_GATE_ANNRET
        )

        coverage_df.to_csv(paths["coverage_csv"], index=False)
        _write_csv(paths["candidates_csv"], candidate_rows)
        _write_csv(paths["validation_selection_csv"], valid_bt_rows)

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
        _write_csv(paths["split_metrics_csv"], split_rows)

        pred_test = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
        with paths["candidate_pred_pkl"].open("wb") as f:
            pickle.dump(pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_test.reset_index().to_csv(paths["candidate_pred_csv"], index=False)

        summary.update(
            {
                "status": "ok",
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "training_protocol": {
                    "raw_start": RAW_START,
                    "model_family": "closed_form_ridge_rank",
                    "feature_source": "local QuantMaster CN data store OHLCV/factor bins via pre2024_train_new_model_lockstep helpers",
                    "feature_modes": list(selected_feature_modes.keys()),
                    "target_modes": target_modes,
                    "alpha_grid": alpha_grid,
                    "selection_rule": "2023-only real net-cost backtest max IR, tie by AnnRet; no 2024-2026 metric used for selection",
                    "preselection_rule": "2023 daily rank IC IR shortlist before 2023 backtests",
                    "preselected_candidates": [r["candidate_id"] for r in preselected],
                    "topk_grid": topk_grid,
                    "ndrop_grid": ndrop_grid,
                    "smoke": bool(args.smoke),
                },
                "splits": {
                    "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": int(len(train_df))},
                    "valid": {"start": VALID_START, "end": VALID_END, "rows": int(len(valid_df))},
                    "test": {"start": TEST_START, "end": TEST_END, "rows": int(len(test_df))},
                },
                "coverage": {
                    "instrument_count": int(coverage_df["instrument"].nunique()),
                    "coverage_csv": str(paths["coverage_csv"]),
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
                    "valid_2023_selection_only": {
                        "rank_ic": selected_candidate["valid_rank_ic"],
                        "rank_ic_ir": selected_candidate["valid_rank_ic_ir"],
                        "costed_ir": selected["costed_ir"],
                        "costed_annret": selected["costed_annret"],
                        "max_drawdown": selected["max_drawdown"],
                        "turnover": selected["turnover"],
                        "error": selected["error"],
                    },
                    "test_2024_2026_hard_gate": {
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
                    "rule": {
                        "scope": "test_2024_01_01_to_2026_04_30_only",
                        "ir_gt": HARD_GATE_IR,
                        "annret_gt": HARD_GATE_ANNRET,
                        "open_cost": float(args.open_cost),
                        "close_cost": float(args.close_cost),
                    },
                    "passed": hard_gate_pass,
                },
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Compact Linear Rank Pre-2024 V2 ({stamp})",
                    "",
                    f"- hard_gate_pass: `{hard_gate_pass}`",
                    f"- selected: `{selected['candidate_id']}` topk/n_drop `{selected['topk']}/{selected['n_drop']}`",
                    f"- valid_2023 selection IR/AnnRet: `{_safe_float(selected['costed_ir']):.6f}` / `{_safe_float(selected['costed_annret']):.6f}`",
                    f"- test_2024_2026 hard-gate IR/AnnRet: `{_safe_float(test_metric.ir):.6f}` / `{_safe_float(test_metric.annret):.6f}`",
                    f"- costs: open `{args.open_cost}` close `{args.close_cost}`",
                    f"- protocol: train `{TRAIN_START}..{TRAIN_END}`, select `{VALID_START}..{VALID_END}`, test once `{TEST_START}..{TEST_END}`",
                    f"- artifacts: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        smoke = {
            "summary_json_exists": paths["summary_json"].exists(),
            "summary_md_exists": paths["summary_md"].exists(),
            "coverage_csv_rows": int(len(pd.read_csv(paths["coverage_csv"]))) if paths["coverage_csv"].exists() else 0,
            "candidates_csv_rows": int(len(pd.read_csv(paths["candidates_csv"]))) if paths["candidates_csv"].exists() else 0,
            "validation_selection_csv_rows": int(len(pd.read_csv(paths["validation_selection_csv"])))
            if paths["validation_selection_csv"].exists()
            else 0,
            "split_metrics_csv_rows": int(len(pd.read_csv(paths["split_metrics_csv"])))
            if paths["split_metrics_csv"].exists()
            else 0,
            "candidate_pred_rows": int(len(_load_pickle(paths["candidate_pred_pkl"])))
            if paths["candidate_pred_pkl"].exists()
            else 0,
            "hard_gate_passed": hard_gate_pass,
            "hard_gate_scope": "test_2024_01_01_to_2026_04_30_only",
        }
        paths["artifact_parse_smoke_json"].write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        _write_failure(paths, summary, exc, t0_all)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

