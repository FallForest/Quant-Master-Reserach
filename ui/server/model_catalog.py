"""模型目录：从 models.yaml 加载，为所有端点提供统一数据源。"""
import yaml
from pathlib import Path

_MODELS_FILE = Path(__file__).resolve().parent.parent / "models.yaml"

# 模型 ID → benchmark YAML config 路径（相对 examples/benchmarks/）
_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "benchmarks"
_BENCH_YAML_MAP = {
    "lightgbm":     "LightGBM/workflow_config_lightgbm_Alpha158.yaml",
    "xgboost":      "XGBoost/workflow_config_xgboost_Alpha158.yaml",
    "catboost":     "CatBoost/workflow_config_catboost_Alpha158.yaml",
    "topk_metalabel": "Transcendence/workflow_config_transcendence_metalabel_topbottom_moderate_Alpha158_2026_csi300.yaml",
    "hflgb":        "LightGBM/workflow_config_lightgbm_Alpha158.yaml",
    "lstm":         "LSTM/workflow_config_lstm_Alpha158.yaml",
    "gru":          "GRU/workflow_config_gru_Alpha158.yaml",
    "alstm":        "ALSTM/workflow_config_alstm_Alpha158.yaml",
    "adarnn":       "ADARNN/workflow_config_adarnn_Alpha360.yaml",
    "krnn":         "KRNN/workflow_config_krnn_Alpha360.yaml",
    "tcts":         "TCTS/workflow_config_tcts_Alpha360.yaml",
    "transformer":  "Transformer/workflow_config_transformer_Alpha158.yaml",
    "gats":         "GATs/workflow_config_gats_Alpha360.yaml",
    "tcn":          "TCN/workflow_config_tcn_Alpha158.yaml",
    "tabnet":       "TabNet/workflow_config_tabnet_Alpha158.yaml",
    "dnn":          "DNN/workflow_config_dnn_Alpha158.yaml",
    "tft":          "TFT/workflow_config_tft_Alpha360.yaml",
    "sfm":          "SFM/workflow_config_sfm_Alpha360.yaml",
    "add":          "ADD/workflow_config_add_Alpha360.yaml",
    "linear":       "Linear/workflow_config_linear_Alpha158.yaml",
    "double_ensemble": "DoubleEnsemble/workflow_config_double_ensemble_Alpha158.yaml",
    "tree_cn_lstm_rl": "Transcendence/workflow_config_tree_cn_lstm_rl_moderate_Alpha158_2026_csi300.yaml",
    "de_residual_lstm": "Transcendence/workflow_config_de_residual_cn_lstm_moderate_Alpha158_2026_csi300.yaml",
    "adaptive_ensemble": "Transcendence/workflow_config_adaptive_ensemble_moderate_Alpha158_2026_csi300.yaml",
    "meta_ensemble": "Transcendence/workflow_config_meta_ensemble_moderate_Alpha158_2026_csi300.yaml",
    "dynamic_meta_ensemble": "Transcendence/workflow_config_dynamic_meta_ensemble_moderate_Alpha158_2026_csi300.yaml",
    "multiseed_de": "Transcendence/workflow_config_multiseed_de_moderate_Alpha158_2026_csi300.yaml",
    "cost_aware_de": "Transcendence/workflow_config_cost_aware_de_moderate_Alpha158_2026_csi300.yaml",
    "low_turnover_de": "Transcendence/workflow_config_low_turnover_de_moderate_Alpha158_2026_csi300.yaml",
    "residual_de": "Transcendence/workflow_config_residual_de_moderate_Alpha158_2026_csi300.yaml",
    "localformer":  "LocalFormer/workflow_config_localformer_Alpha360.yaml",
    "hist":         "HIST/workflow_config_hist_Alpha360.yaml",
    "igmtf":        "IGMTF/workflow_config_igmtf_Alpha360.yaml",
    "sandwich":     "Sandwich/workflow_config_sandwich_Alpha360.yaml",
    "tra":          "TRA/workflow_config_tra_Alpha360.yaml",
    "general_ptnn": "GeneralPTNN/workflow_config_general_ptnn_Alpha158.yaml",
    "pretrained_signal": "Transcendence/workflow_config_doubleensemble_baseline_repro_Alpha158_2026_csi300.yaml",
    "regime_horizon_cost_ensemble": "Transcendence/workflow_config_regime_horizon_cost_moderate_Alpha158_2026_csi300.yaml",
    "transcendence_hybrid": "Transcendence/workflow_config_transcendence_hybrid_moderate_Alpha158_2026_csi300.yaml",
    "transcendence_signal": "Transcendence/workflow_config_doubleensemble_baseline_repro_Alpha158_2026_csi300.yaml",
}

# 默认策略参数
DEFAULT_STRATEGY_PARAMS = {
    "topk": 50, "n_drop": 5,
    "open_cost": 0.0005, "close_cost": 0.0015,
    "limit_threshold": 0.095, "deal_price": "close",
}


def _load_catalog():
    """加载 models.yaml，返回 (models_dict, handlers_dict)。"""
    with open(_MODELS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {}), data.get("handlers", {})


_MODELS, _HANDLERS = _load_catalog()


def get_models():
    """返回模型字典 {id: spec}。"""
    return _MODELS


def get_handlers():
    """返回处理器字典 {id: spec}。"""
    return _HANDLERS


def get_bench_yaml_path(model_id):
    """返回模型对应的 benchmark YAML 绝对路径，找不到返回 None。"""
    rel = _BENCH_YAML_MAP.get(model_id)
    if rel:
        p = _BENCH_DIR / rel
        return str(p) if p.exists() else None
    return None


def get_strategy_defaults(model_id):
    """返回模型的推荐策略参数。"""
    spec = _MODELS.get(model_id, {})
    overrides = spec.get("strategy_overrides", {})
    return {**DEFAULT_STRATEGY_PARAMS, **overrides}


def get_model_catalog_list():
    """返回模型工坊所需的完整模型列表（含 params/complexity/speed/icon/strategyDefaults）。"""
    result = []
    for mid, spec in _MODELS.items():
        entry = {
            "id": mid,
            "name": spec["label"],
            "category": spec["category"],
            "desc": spec.get("desc", ""),
            "icon": spec.get("icon", "dnn"),
            "params": spec.get("params", {}),
            "complexity": spec.get("complexity", 3),
            "speed": spec.get("speed", 3),
            "strategyDefaults": get_strategy_defaults(mid),
        }
        result.append(entry)
    return result


def get_model_stock_select_list():
    """返回选股页面所需的模型列表（含 id/label/category/desc/handler）。"""
    result = []
    for mid, spec in _MODELS.items():
        result.append({
            "id": mid,
            "label": spec["label"],
            "category": spec["category"],
            "desc": spec.get("desc", ""),
            "handler": spec["handler"],
        })
    return result


def get_categories():
    """返回所有模型分类列表。"""
    cats = []
    for spec in _MODELS.values():
        c = spec.get("category", "")
        if c and c not in cats:
            cats.append(c)
    return cats
