"""静态数据：模型列表、模型目录、策略列表、优化器。"""
from .. import app
from ..model_catalog import (
    get_handlers as _get_handlers,
    get_model_catalog_list, get_model_stock_select_list, get_categories,
)


def models(rh):
    models_list = get_model_stock_select_list()
    handlers_list = [{"id": k, "label": v["label"]} for k, v in _get_handlers().items()]
    rh._json_response({"models": models_list, "handlers": handlers_list})


def model_catalog(rh):
    model_list = get_model_catalog_list()
    categories = get_categories()
    rh._json_response({"models": model_list, "categories": categories})


def strategies(rh):
    """返回策略列表，performance 从实际回测结果中提取。"""
    perf = {"avgReturn": None, "sharpe": None, "maxDD": None, "turnover": None}
    bt_runs = getattr(app, 'backtest_runs', {})
    for run_id in sorted(bt_runs.keys(), reverse=True):
        run = bt_runs.get(run_id, {})
        if run.get("done") and run.get("success") and run.get("results"):
            m = run["results"].get("metrics", {})
            if m:
                perf = {
                    "avgReturn": m.get("annualReturn"),
                    "sharpe": m.get("sharpe"),
                    "maxDD": m.get("maxDrawdown"),
                    "turnover": m.get("avgTurnover"),
                }
            break
    strategy_list = [
        {
            "id": "topk_dropout", "name": "TopK Dropout", "category": "选股",
            "desc": "经典Top-K选股策略，每期淘汰底部N只，买入顶部N只，适合高换手Alpha信号",
            "params": {
                "topk": {"label": "Top K", "type": "number", "default": 50, "min": 5, "max": 200, "desc": "持有股票数量"},
                "n_drop": {"label": "Dropout N", "type": "number", "default": 5, "min": 1, "max": 50, "desc": "每期换入/换出数量"},
                "method_sell": {"label": "卖出方式", "type": "select", "options": ["sell_first", "sell_by_amount"], "default": "sell_first", "desc": "卖出优先级"},
                "method_buy": {"label": "买入方式", "type": "select", "options": ["buy_first", "buy_by_amount"], "default": "buy_first", "desc": "买入优先级"},
                "hold_thresh": {"label": "持有阈值", "type": "number", "default": 0, "min": 0, "max": 100, "desc": "最小持有天数"},
            },
            "performance": perf,
        },
        {
            "id": "soft_topk", "name": "Soft TopK", "category": "选股",
            "desc": "成本感知的Top-K策略，通过trade_impact_limit控制单期换手幅度，降低交易成本",
            "params": {
                "topk": {"label": "Top K", "type": "number", "default": 30, "min": 5, "max": 200, "desc": "持有股票数量"},
                "trade_impact_limit": {"label": "换手上限", "type": "number", "default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01, "desc": "单期最大权重变动"},
                "selection_rank_buffer": {"label": "排名缓冲", "type": "number", "default": 20, "min": 0, "max": 100, "desc": "候选池扩展范围"},
                "risk_degree": {"label": "风险度", "type": "number", "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "desc": "风险敞口比例"},
                "buy_method": {"label": "买入方式", "type": "select", "options": ["proportion", "equal"], "default": "proportion", "desc": "资金分配方式"},
            },
            "performance": perf,
        },
        {
            "id": "enhanced_indexing", "name": "增强指数", "category": "增强",
            "desc": "指数增强策略，在跟踪基准的同时最大化Alpha收益，支持因子暴露约束和跟踪误差控制",
            "params": {
                "riskmodel_root": {"label": "风险模型", "type": "text", "default": "", "desc": "风险模型数据路径"},
                "market": {"label": "市场", "type": "select", "options": ["cn", "us"], "default": "cn", "desc": "市场选择"},
                "turn_limit": {"label": "换手限制", "type": "number", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05, "desc": "最大换手率"},
            },
            "performance": perf,
        },
        {
            "id": "twap", "name": "TWAP 执行", "category": "执行",
            "desc": "时间加权平均价格执行策略，将订单均匀分配到多个时间切片，降低市场冲击",
            "params": {
                "freq": {"label": "频率", "type": "select", "options": ["day", "1min", "5min"], "default": "day", "desc": "执行频率"},
            },
            "performance": {"avgReturn": None, "sharpe": None, "maxDD": None, "turnover": None},
        },
        {
            "id": "sbb_ema", "name": "SBB EMA 择时", "category": "执行",
            "desc": "基于EMA趋势判断的择时执行策略，在趋势向好时加仓、向差时减仓",
            "params": {
                "instruments": {"label": "标的列表", "type": "text", "default": "", "desc": "逗号分隔的股票代码"},
                "freq": {"label": "频率", "type": "select", "options": ["day", "1min"], "default": "day", "desc": "执行频率"},
            },
            "performance": perf,
        },
        {
            "id": "ac_strategy", "name": "AC 波动执行", "category": "执行",
            "desc": "基于Calmar比率和波动率自适应的执行策略，用双曲正弦函数动态调整下单节奏",
            "params": {
                "lamb": {"label": "Lambda", "type": "number", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "desc": "风险厌恶系数"},
                "eta": {"label": "Eta", "type": "number", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1, "desc": "自适应步长"},
                "window_size": {"label": "窗口大小", "type": "number", "default": 20, "min": 5, "max": 100, "desc": "波动率计算窗口"},
            },
            "performance": perf,
        },
    ]
    rh._json_response({"strategies": strategy_list})


def optimizer(rh):
    methods = [
        {
            "id": "gmv", "name": "全局最小方差 (GMV)",
            "desc": "不考虑预期收益，仅最小化组合方差，适合对收益预测不自信时使用",
            "params": {
                "delta": {"label": "换手限制", "type": "number", "default": 0.1, "min": 0, "max": 1.0, "step": 0.01, "desc": "最大单期换手率"},
                "alpha": {"label": "L2正则", "type": "number", "default": 0.01, "min": 0, "max": 1.0, "step": 0.001, "desc": "权重L2正则化系数"},
            },
        },
        {
            "id": "mvo", "name": "均值方差优化 (MVO)",
            "desc": "经典Markowitz优化，在风险-收益之间取得最优平衡",
            "params": {
                "lamb": {"label": "风险厌恶系数", "type": "number", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "desc": "越大越保守"},
                "delta": {"label": "换手限制", "type": "number", "default": 0.1, "min": 0, "max": 1.0, "step": 0.01, "desc": "最大单期换手率"},
                "alpha": {"label": "L2正则", "type": "number", "default": 0.01, "min": 0, "max": 1.0, "step": 0.001, "desc": "权重L2正则化系数"},
                "scale_return": {"label": "收益缩放", "type": "number", "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1, "desc": "预期收益缩放因子"},
            },
        },
        {
            "id": "rp", "name": "风险平价 (RP)",
            "desc": "各资产对组合风险的贡献相等，适合多因子/多资产场景",
            "params": {
                "delta": {"label": "换手限制", "type": "number", "default": 0.1, "min": 0, "max": 1.0, "step": 0.01, "desc": "最大单期换手率"},
                "alpha": {"label": "L2正则", "type": "number", "default": 0.01, "min": 0, "max": 1.0, "step": 0.001, "desc": "权重L2正则化系数"},
            },
        },
        {
            "id": "inv_vol", "name": "逆波动率 (InvVol)",
            "desc": "按波动率倒数分配权重，波动率越低权重越高，简单稳健",
            "params": {
                "delta": {"label": "换手限制", "type": "number", "default": 0.1, "min": 0, "max": 1.0, "step": 0.01, "desc": "最大单期换手率"},
            },
        },
        {
            "id": "enhanced_indexing", "name": "增强指数优化",
            "desc": "CVXPY求解的增强指数优化器，支持因子暴露约束、跟踪误差限制、强制持仓等",
            "params": {
                "lamb": {"label": "风险厌恶系数", "type": "number", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "desc": "越大越保守"},
                "delta": {"label": "换手限制", "type": "number", "default": 0.08, "min": 0, "max": 1.0, "step": 0.01, "desc": "最大单期换手率"},
                "b_dev": {"label": "基准偏离", "type": "number", "default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001, "desc": "最大基准偏离度"},
                "f_dev": {"label": "因子偏离", "type": "number", "default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001, "desc": "最大因子偏离度"},
            },
        },
    ]
    rh._json_response({"methods": methods, "comparison": [], "sectors": []})
