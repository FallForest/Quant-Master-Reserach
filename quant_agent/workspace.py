from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from quant_agent.templates import FACTOR_WORKFLOW_TEMPLATE, MODEL_WORKFLOW_TEMPLATE

_jinja_env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CN_DATA_DIR = str(
    Path(os.getenv("QM_CN_DATA_DIR", _PROJECT_ROOT.joinpath(".qmData", "cn_data"))).expanduser().resolve().as_posix()
)

ALPHA20: dict[str, str] = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}


def create_workspace(
    root: Path,
    action: str,
    hypothesis_payload: dict[str, Any],
    experiment_payload: dict[str, Any],
    prompt_dump: dict[str, str],
    model_code_payload: dict[str, Any] | None = None,
    quick_smoke: bool = False,
    workflow_overrides: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    (root / "prompts").mkdir(exist_ok=True)
    for name, content in prompt_dump.items():
        (root / "prompts" / f"{name}.txt").write_text(content, encoding="utf-8")

    (root / "hypothesis.json").write_text(
        json.dumps(hypothesis_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "experiment.json").write_text(
        json.dumps(experiment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if action == "factor":
        _write_factor_runtime_files(root, experiment_payload, quick_smoke=quick_smoke, workflow_overrides=workflow_overrides)
    elif action == "model":
        _write_model_runtime_files(root, experiment_payload, model_code_payload, quick_smoke=quick_smoke, workflow_overrides=workflow_overrides)
    (root / "run_experiment.bat").write_text(_build_batch_file(action), encoding="utf-8")
    return root


def _build_batch_file(action: str) -> str:
    if action == "factor":
        config_path = "rendered_factor_workflow.yaml"
    else:
        config_path = "rendered_model_workflow.yaml"
    repo_root = str(_PROJECT_ROOT)
    return "\n".join(
        [
            "@echo off",
            "setlocal",
            "cd /d %~dp0",
            "echo Running Quant-Master workflow from %CD%",
            f'set "REPO_ROOT={repo_root}"',
            'if exist "%REPO_ROOT%\\.venv\\Scripts\\python.exe" (set "QPY=%REPO_ROOT%\\.venv\\Scripts\\python.exe") else (set "QPY=python")',
            "set PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%",
            "set OMP_NUM_THREADS=4",
            "set OPENBLAS_NUM_THREADS=4",
            "set MKL_NUM_THREADS=4",
            "set NUMEXPR_NUM_THREADS=4",
            f'%QPY% -m quant_master.cli.run "%CD%\\{config_path}"',
            "endlocal",
            "",
        ]
    )


def _write_factor_runtime_files(root: Path, experiment_payload: dict[str, Any], quick_smoke: bool = False, workflow_overrides: dict[str, Any] | None = None) -> None:
    if experiment_payload:
        feature_names = list(experiment_payload.keys())
        feature_expressions = [_formula_to_expression(str(spec.get("formulation", ""))) for spec in experiment_payload.values()]
    else:
        feature_names = list(ALPHA20.keys())
        feature_expressions = list(ALPHA20.values())

    context = {
        **_rd_time_context(quick_smoke=quick_smoke),
        "feature_names": str(feature_names),
        "feature_expressions": str(feature_expressions),
    }
    rendered = _render_template_string(FACTOR_WORKFLOW_TEMPLATE, context)
    rendered = _inject_provider_uri(rendered)
    if quick_smoke:
        rendered = _apply_quick_smoke_workflow_tuning(rendered)
    if workflow_overrides:
        rendered = _apply_workflow_overrides(rendered, workflow_overrides)
    (root / "rendered_factor_workflow.yaml").write_text(rendered, encoding="utf-8")


def _write_model_runtime_files(
    root: Path,
    experiment_payload: dict[str, Any],
    model_code_payload: dict[str, Any] | None,
    quick_smoke: bool = False,
    workflow_overrides: dict[str, Any] | None = None,
) -> None:
    if model_code_payload is None:
        raise RuntimeError("Model workspace requires generated model code.")
    model_code = _patch_model_code_for_quant_master(str(model_code_payload["code"]))
    (root / "model.py").write_text(model_code, encoding="utf-8")

    model_name, spec = next(iter(experiment_payload.items()))
    training = dict(spec.get("training_hyperparameters", {}))
    training = _apply_quick_smoke_training(training) if quick_smoke else training
    model_type = str(spec.get("model_type", "Tabular"))
    feature_names = list(ALPHA20.keys())
    feature_expressions = list(ALPHA20.values())
    if model_type.lower() == "timeseries":
        dataset_cls = "TSDatasetH"
        step_len = 20
        num_timesteps = 20
    else:
        dataset_cls = "DatasetH"
        step_len = None
        num_timesteps = None
    context = {
        **_rd_time_context(quick_smoke=quick_smoke),
        "feature_names": str(feature_names),
        "feature_expressions": str(feature_expressions),
        "n_epochs": int(training.get("n_epochs", 100)),
        "lr": float(training.get("lr", 2e-4)),
        "early_stop": int(training.get("early_stop", 10)),
        "batch_size": int(training.get("batch_size", 4096)),
        "weight_decay": float(training.get("weight_decay", 0.0001)),
        "dataset_cls": dataset_cls,
        "num_features": len(ALPHA20),
        "step_len": step_len,
        "num_timesteps": num_timesteps,
    }
    rendered = _render_template_string(MODEL_WORKFLOW_TEMPLATE, context)
    rendered = _inject_provider_uri(rendered)
    rendered = _normalize_timeseries_model_workflow(rendered)
    if quick_smoke:
        rendered = _apply_quick_smoke_workflow_tuning(rendered)
    if workflow_overrides:
        rendered = _apply_workflow_overrides(rendered, workflow_overrides)

    (root / "model_spec.json").write_text(
        json.dumps({model_name: spec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "rendered_model_workflow.yaml").write_text(rendered, encoding="utf-8")


def _patch_model_code_for_quant_master(code: str) -> str:
    if "_quant_agent_base_model_cls" in code and "model_cls = QuantMasterModelWrapper" in code:
        return code
    if "model_cls =" not in code:
        return code
    code = code.replace("model_cls =", "_quant_agent_base_model_cls =")
    wrapper = """


class QuantMasterModelWrapper(_quant_agent_base_model_cls):
    def __init__(self, num_features=None, num_timesteps=None, input_dim=None, **kwargs):
        if input_dim is None:
            input_dim = num_features
        if input_dim is None:
            raise ValueError("Either input_dim or num_features must be provided.")
        super().__init__(input_dim=input_dim, **kwargs)


model_cls = QuantMasterModelWrapper
"""
    return code.rstrip() + wrapper


def _formula_to_expression(formulation: str) -> str:
    compact = formulation.replace(" ", "")
    lowered = compact.lower()

    if "close_t" in lowered and "close_{t-5}" in lowered and ("\\frac" in compact or "/" in compact):
        return "$close/Ref($close,5)-1"

    if "volume_t" in lowered and "volume_{t-5}" in lowered and ("\\frac" in compact or "/" in compact):
        return "$volume/Ref($volume,5)-1"

    if "rv_{10}" in lowered or ("std" in lowered and "close_t" in lowered and "close_{t-1}" in lowered):
        return "Std($close/Ref($close,1)-1,10)"

    if "rv_{20}" in lowered:
        return "Std($close/Ref($close,1)-1,20)"

    if "turnoverratio" in lowered:
        return "Mean($turnover,10)"

    if "momentum" in lowered:
        return "$close/Ref($close,5)-1"
    if "volatility" in lowered:
        return "Std($close/Ref($close,1)-1,10)"
    if "volume" in lowered:
        return "$volume/Ref($volume,5)-1"
    return "$close/$close"


def _rd_default_time_context() -> dict[str, str]:
    return {
        "train_start": "2008-01-01",
        "train_end": "2014-12-31",
        "valid_start": "2015-01-01",
        "valid_end": "2016-12-31",
        "test_start": "2017-01-01",
        "test_end": "2020-08-01",
    }


def _rd_quick_smoke_time_context() -> dict[str, str]:
    return {
        "train_start": "2019-01-01",
        "train_end": "2019-06-30",
        "valid_start": "2019-07-01",
        "valid_end": "2019-09-30",
        "test_start": "2019-10-01",
        "test_end": "2019-12-31",
    }


def _rd_time_context(quick_smoke: bool = False) -> dict[str, str]:
    return _rd_quick_smoke_time_context() if quick_smoke else _rd_default_time_context()


def _apply_quick_smoke_training(training: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(training)
    tuned["n_epochs"] = min(max(int(tuned.get("n_epochs", 100)), 1), 2)
    tuned["early_stop"] = min(max(int(tuned.get("early_stop", 10)), 1), 1)
    tuned["batch_size"] = min(max(int(tuned.get("batch_size", 4096)), 1024), 8192)
    return tuned


def _apply_quick_smoke_workflow_tuning(workflow_text: str) -> str:
    workflow_text = re.sub(r"(?m)^(\s*)n_jobs:\s*\d+\s*$", r"\1n_jobs: 1", workflow_text)
    workflow_text = re.sub(r"(?m)^(\s*)topk:\s*\d+\s*$", r"\1topk: 10", workflow_text)
    workflow_text = re.sub(r"(?m)^(\s*)n_drop:\s*\d+\s*$", r"\1n_drop: 1", workflow_text)
    return workflow_text if workflow_text.endswith("\n") else workflow_text + "\n"


def _apply_workflow_overrides(workflow_text: str, overrides: dict[str, Any]) -> str:
    _YAML_FIELD_MAP = {
        "topk": (r"(?m)^(\s*)topk:\s*[\d.]+\s*$", r"\1topk: {val}"),
        "n_drop": (r"(?m)^(\s*)n_drop:\s*[\d.]+\s*$", r"\1n_drop: {val}"),
        "open_cost": (r"(?m)^(\s*)open_cost:\s*[\d.]+\s*$", r"\1open_cost: {val}"),
        "close_cost": (r"(?m)^(\s*)close_cost:\s*[\d.]+\s*$", r"\1close_cost: {val}"),
        "min_cost": (r"(?m)^(\s*)min_cost:\s*[\d.]+\s*$", r"\1min_cost: {val}"),
        "limit_threshold": (r"(?m)^(\s*)limit_threshold:\s*[\d.]+\s*$", r"\1limit_threshold: {val}"),
    }
    for key, value in overrides.items():
        if key not in _YAML_FIELD_MAP:
            continue
        pattern, replacement_tpl = _YAML_FIELD_MAP[key]
        replacement = replacement_tpl.format(val=value)
        workflow_text = re.sub(pattern, replacement, workflow_text)
    return workflow_text if workflow_text.endswith("\n") else workflow_text + "\n"


def _render_template_string(template_text: str, context: dict[str, Any]) -> str:
    return _jinja_env.from_string(template_text).render(**context).strip() + "\n"


def _inject_provider_uri(workflow_text: str) -> str:
    lines = workflow_text.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^\s*provider_uri\s*:", line):
            indent = line[: len(line) - len(line.lstrip())]
            lines[idx] = f'{indent}provider_uri: "{_DEFAULT_CN_DATA_DIR}"'
            break
    rendered = "\n".join(lines)
    return rendered if rendered.endswith("\n") else rendered + "\n"


def _normalize_timeseries_model_workflow(workflow_text: str) -> str:
    workflow_text = re.sub(
        r'(?ms)(^\s*pt_model_kwargs:)\s*\{\s*"num_features":\s*(\d+)\s*,\s*\}',
        r"\1\n              num_features: \2",
        workflow_text,
    )
    workflow_text = re.sub(
        r'(?ms)(^\s*pt_model_kwargs:)\s*\{\s*"num_features":\s*(\d+),\s*num_timesteps:\s*(\d+)\s*\}',
        r"\1\n              num_features: \2\n              num_timesteps: \3",
        workflow_text,
    )
    workflow_text = re.sub(
        r"(?m)^step_len:\s*(\d+)\s+record:\s*$",
        r"            step_len: \1\n    record:",
        workflow_text,
    )
    return workflow_text if workflow_text.endswith("\n") else workflow_text + "\n"
