"""Run purged OOF stacking with data/universe preflight diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

import quant_master
from quant_master.cli.run import render_template
from quant_master.contrib.data.amount_flow_handler import Alpha158AmountFlow, Alpha158AmountFlowCore, Alpha360TDX
from quant_master.data import D
from quant_master.model.trainer import task_train
from quant_master.utils.data import update_config
from quant_master.workflow import R


DEFAULT_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "workflows"
    / "ensemble"
    / "workflow_config_purged_oof_stacking_Alpha168_tdx_personal_under40_v1.yaml"
)


def _format_diagnostics(diagnostics):
    return json.dumps(diagnostics, indent=2, sort_keys=True, default=str)


def _load_config(config_path, seen=None):
    config_path = Path(config_path).resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        raise ValueError(f"Circular BASE_CONFIG_PATH reference: {config_path}")
    seen.add(config_path)
    config = YAML(typ="safe", pure=True).load(render_template(str(config_path)))
    base_config_path = config.pop("BASE_CONFIG_PATH", None)
    if not base_config_path:
        return config
    base_path = Path(base_config_path)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_config = _load_config(base_path, seen=seen)
    return update_config(base_config, config)


def _run_preflight(config):
    preflight = config.get("data_preflight")
    if not preflight:
        return {}
    market = preflight["market"]
    filter_pipe = preflight.get("filter_pipe", [])
    filtered_market = D.instruments(market, filter_pipe=filter_pipe)
    anchors = {}
    eligible_anchors = {}
    eligible_ratios = {}
    coverage = {}
    provider_uri = Path(config["quant_master_init"]["provider_uri"]).expanduser()
    feature_root = provider_uri / "features"
    for date in preflight["anchor_dates"]:
        instruments = D.list_instruments(D.instruments(market), start_time=date, end_time=date, as_list=True)
        count = len(instruments)
        if not preflight["min_universe_size"] <= count <= preflight["max_universe_size"]:
            raise ValueError(f"Universe preflight failed on {date}: {market} has {count} instruments.")
        eligible = D.list_instruments(filtered_market, start_time=date, end_time=date, as_list=True)
        eligible_count = len(eligible)
        min_eligible = int(preflight.get("min_eligible_size", 0))
        max_eligible = int(preflight.get("max_eligible_size", count))
        if not min_eligible <= eligible_count <= max_eligible:
            raise ValueError(
                f"Eligible universe preflight failed on {date}: {market} has {eligible_count} instruments."
            )
        eligible_ratio = eligible_count / max(count, 1)
        min_ratio = float(preflight.get("min_eligible_ratio", 0.0))
        max_ratio = float(preflight.get("max_eligible_ratio", 1.0))
        if not min_ratio <= eligible_ratio <= max_ratio:
            raise ValueError(
                f"Eligible universe ratio preflight failed on {date}: {eligible_ratio:.4f}."
            )
        anchors[str(date)] = count
        eligible_anchors[str(date)] = eligible_count
        eligible_ratios[str(date)] = eligible_ratio
        covered = sum((feature_root / instrument.lower()).is_dir() for instrument in eligible)
        ratio = covered / max(eligible_count, 1)
        if ratio < float(preflight["min_feature_dir_coverage"]):
            raise ValueError(f"Feature directory coverage preflight failed on {date}: {ratio:.4f}.")
        coverage[str(date)] = ratio

    sample_date = str(preflight["feature_sample_end"])
    sample_instruments = D.list_instruments(
        filtered_market, start_time=sample_date, end_time=sample_date, as_list=True
    )[: int(preflight["feature_sample_size"])]
    feature_handlers = {
        "alpha360_tdx": Alpha360TDX,
        "alpha158_amount_flow": Alpha158AmountFlow,
        "alpha158_amount_flow_core": Alpha158AmountFlowCore,
    }
    feature_set = preflight.get("feature_set", "alpha360_tdx")
    if feature_set not in feature_handlers:
        raise ValueError(f"Unsupported preflight feature_set: {feature_set}")
    handler_class = feature_handlers[feature_set]
    handler = handler_class(init_data=False)
    fields, names = handler.get_feature_config()
    features = D.features(
        sample_instruments,
        fields,
        start_time=str(preflight["feature_sample_start"]),
        end_time=sample_date,
        freq="day",
    ).replace([np.inf, -np.inf], np.nan)
    expected_columns = int(preflight["expected_feature_columns"])
    if features.shape[1] != expected_columns or len(names) != expected_columns:
        raise ValueError(f"Feature preflight expected {expected_columns} columns, got {features.shape[1]}.")
    missing_columns = [names[index] for index, column in enumerate(features.columns) if features[column].notna().sum() == 0]
    if missing_columns:
        raise ValueError(f"TDX Alpha360 preflight found all-missing columns: {missing_columns[:10]}.")
    finite_ratio = float(features.notna().to_numpy().mean())
    if finite_ratio < 0.95:
        raise ValueError(f"TDX Alpha360 preflight finite ratio {finite_ratio:.4f} < 0.95.")
    return {
        "market": market,
        "anchor_counts": anchors,
        "eligible_anchor_counts": eligible_anchors,
        "eligible_anchor_ratios": eligible_ratios,
        "filter_pipe": filter_pipe,
        "feature_dir_coverage": coverage,
        "feature_rows": len(features),
        "feature_columns": features.shape[1],
        "feature_finite_ratio": finite_ratio,
        "feature_sample_instruments": sample_instruments,
    }


def _held_out_test_diagnostics(model, recorder, segment=None):
    prediction = recorder.load_object("pred.pkl").iloc[:, 0]
    label = recorder.load_object("label.pkl").iloc[:, 0]
    frame = prediction.rename("prediction").to_frame()
    frame["label"] = label.reindex(frame.index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if segment is not None:
        start, end = map(pd.Timestamp, segment)
        dates = frame.index.get_level_values("datetime")
        frame = frame.loc[(dates >= start) & (dates <= end)]
    if frame.empty:
        raise ValueError("Held-out diagnostics found no aligned finite rows in the configured test segment.")
    prediction = frame["prediction"]
    label = frame["label"]
    overall = model._rank_ic_summary(prediction, label)
    years = prediction.index.get_level_values("datetime").year
    yearly = {
        str(year): model._rank_ic_summary(prediction.loc[years == year], label.loc[years == year])
        for year in sorted(set(years))
    }
    daily_rows = frame.groupby(level="datetime").size()
    coverage = {
        "min_daily_rows": int(daily_rows.min()),
        "median_daily_rows": float(daily_rows.median()),
        "max_daily_rows": int(daily_rows.max()),
        "min_daily_rows_date": str(daily_rows.idxmin().date()),
    }
    return {"overall": overall, "yearly": yearly, "coverage": coverage}


def _assert_held_out_gate(diagnostics, gate):
    failures = []
    overall = diagnostics["overall"]
    if overall["rank_ic"] < float(gate["min_rank_ic"]):
        failures.append(f"RankIC {overall['rank_ic']:.6f} < {float(gate['min_rank_ic']):.6f}")
    if overall["rank_icir"] < float(gate["min_rank_icir"]):
        failures.append(f"RankICIR {overall['rank_icir']:.6f} < {float(gate['min_rank_icir']):.6f}")
    weak_years = {
        year: values["rank_ic"]
        for year, values in diagnostics["yearly"].items()
        if values["rank_ic"] < float(gate["min_yearly_rank_ic"])
    }
    if weak_years:
        failures.append(f"yearly RankIC below floor: {weak_years}")
    min_daily_rows = gate.get("min_daily_rows")
    if min_daily_rows is not None and diagnostics["coverage"]["min_daily_rows"] < int(min_daily_rows):
        failures.append(
            f"minimum daily rows {diagnostics['coverage']['min_daily_rows']} < {int(min_daily_rows)}"
        )
    if failures:
        raise ValueError("Held-out prediction gate failed: " + "; ".join(failures))


def _get_recorder(config, recorder_id=None):
    experiment_name = config.get("experiment_name", "purged_oof_stacking")
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    return task_train(config["task"], experiment_name=experiment_name)


def run(config_path=DEFAULT_CONFIG, output_path=None, recorder_id=None):
    config_path = Path(config_path)
    config = _load_config(config_path)
    quant_master.init(**config["quant_master_init"])
    preflight = _run_preflight(config)
    recorder = _get_recorder(config, recorder_id=recorder_id)
    model = recorder.load_object("params.pkl")
    diagnostics = model.get_oof_diagnostics()
    diagnostics["data_preflight"] = preflight
    test_segment = config["task"]["dataset"]["kwargs"]["segments"].get("test")
    diagnostics["held_out_test"] = _held_out_test_diagnostics(model, recorder, segment=test_segment)
    gate = config.get("prediction_gate")
    output_path = Path(output_path) if output_path else config_path.with_suffix(".diagnostics.json")
    formatted_diagnostics = _format_diagnostics(diagnostics)
    output_path.write_text(formatted_diagnostics, encoding="utf-8")
    print(formatted_diagnostics)
    if gate:
        model.assert_oof_gate(**gate)
    held_out_gate = config.get("held_out_prediction_gate")
    if held_out_gate:
        _assert_held_out_gate(diagnostics["held_out_test"], held_out_gate)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--recorder-id", help="Recover diagnostics from a completed MLflow run without retraining.")
    args = parser.parse_args()
    run(args.config, args.output, recorder_id=args.recorder_id)


if __name__ == "__main__":
    main()
