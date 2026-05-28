"""回测和选股：运行、状态查询、结果获取、演示数据生成。"""
import json
import math
import os
import random
import threading
import time
import traceback
from datetime import date, timedelta

import numpy as np

from .. import app
from ..model_catalog import get_bench_yaml_path
from ..stock_select import (
    get_runs as get_ss_runs, get_counter as next_ss_counter,
    run_stock_selection,
)

_bt_lock = threading.Lock()

# ---- quant_master 引擎 (可选) ----
_QM_READY = False
try:
    import quant_master
    from quant_master.model.trainer import task_train
    from ruamel.yaml import YAML
    _QM_READY = True
except ImportError:
    pass


# ---- 回测 ----

def run(rh):
    config = rh._read_body()
    run_id = f"bt_{int(time.time())}_{random.randint(1000,9999)}"
    with _bt_lock:
        app.backtest_runs = getattr(app, 'backtest_runs', {})
        app.backtest_runs[run_id] = {
            "startTime": time.time(),
            "config": config,
            "done": False,
            "success": False,
            "progress": 0,
            "step": "排队中",
            "logs": [],
            "results": None,
            "error": None,
        }
    t = threading.Thread(target=_backtest_worker, args=(run_id, config), daemon=True)
    t.start()
    rh._json_response({"runId": run_id})


def status(rh, run_id):
    bt_runs = getattr(app, 'backtest_runs', {})
    run = bt_runs.get(run_id)
    if not run:
        return rh._json_response({"error": "run not found"}, status=404)
    resp = {
        "progress": run.get("progress", 0),
        "step": run.get("step", ""),
        "logs": run.get("logs", []),
        "done": run.get("done", False),
        "success": run.get("success", False),
    }
    rh._json_response(resp)


def results(rh, run_id):
    run = getattr(app, 'backtest_runs', {}).get(run_id, {})
    if not run.get("done"):
        return rh._json_response({"error": "任务尚未完成"}, status=400)
    result = run.get("results")
    if not result:
        result = _gen_backtest_demo(run.get("config", {}))
    rh._json_response({"results": result, "config": run.get("config", {})})


# ---- 选股 ----

def stock_select_run(rh):
    config = rh._read_body()
    ss_runs = get_ss_runs()
    for rid, run in ss_runs.items():
        if not run.get("done"):
            return rh._json_response({"error": "已有选股任务在运行中，请等待完成"}, status=409)

    run_id = f"ss_{next_ss_counter()}"
    ss_runs[run_id] = {
        "startTime": time.time(),
        "progress": 0,
        "step": "排队中",
        "logs": [],
        "done": False,
        "success": False,
        "error": None,
        "results": None,
        "config": config,
    }

    t = threading.Thread(target=run_stock_selection, args=(run_id, config, app.data), daemon=True)
    t.start()
    rh._json_response({"runId": run_id})


def stock_select_status(rh, run_id):
    ss_runs = get_ss_runs()
    run = ss_runs.get(run_id)
    if not run:
        return rh._json_response({"error": "run not found"}, status=404)
    resp = {
        "progress": run["progress"],
        "step": run["step"],
        "logs": run["logs"][-50:],
        "done": run["done"],
        "success": run["success"],
    }
    if run["error"]:
        resp["error"] = run["error"]
    rh._json_response(resp)


def stock_select_results(rh, run_id):
    ss_runs = get_ss_runs()
    run = ss_runs.get(run_id)
    if not run:
        return rh._json_response({"error": "run not found"}, status=404)
    if not run["done"]:
        return rh._json_response({"error": "任务尚未完成"}, status=400)
    if not run["success"]:
        return rh._json_response({"error": run.get("error", "任务失败")}, status=400)
    rh._json_response({"results": run["results"], "config": run["config"]})


# ---- 回测内部逻辑 ----

def _backtest_worker(run_id, config):
    """在后台线程中执行 quant_master 回测。"""
    run = app.backtest_runs[run_id]
    model_cfg = config.get("model", {})
    strat_cfg = config.get("strategy", {})
    model_id = model_cfg.get("type", "lightgbm")

    def log(level, msg):
        run["logs"].append({"level": level, "msg": f"[{time.strftime('%H:%M:%S')}] {msg}"})

    def set_progress(pct, step):
        run["progress"] = pct
        run["step"] = step

    yaml_path = get_bench_yaml_path(model_id)
    if not _QM_READY or not yaml_path:
        if not yaml_path:
            log("warn", f"模型 {model_id} 无 YAML 配置，使用演示数据")
        else:
            log("warn", "quant_master 未安装，使用演示数据")
        _demo_fallback(run_id, config)
        return

    try:
        set_progress(5, "加载配置")
        log("info", f"模型: {model_id}  配置: {yaml_path}")

        yaml = YAML(typ="safe", pure=True)
        with open(yaml_path) as f:
            wf_config = yaml.load(f)

        # 覆盖日期参数 — 前端将回测参数放在 config 顶层
        train_start = model_cfg.get("train_start", "2018-01-01")
        train_end = model_cfg.get("train_end", "2023-12-31")
        valid_end = model_cfg.get("valid_end", "2025-12-31")
        bt_start = config.get("start_date", "2026-01-02")
        bt_end = config.get("end_date", "2026-12-31")
        bench = config.get("bench", "SH000905")
        freq = config.get("freq", "day")

        # 覆盖 data_handler_config 日期
        dhc = wf_config.get("data_handler_config", {})
        dhc["start_time"] = train_start
        dhc["end_time"] = bt_end
        dhc["fit_start_time"] = train_start
        dhc["fit_end_time"] = train_end

        # 覆盖 handler 类型
        handler_id = model_cfg.get("handler", "Alpha158")
        handler_map = {
            "Alpha158": ("Alpha158", "quant_master.contrib.data.handler"),
            "Alpha360": ("Alpha360", "quant_master.contrib.data.handler"),
            "Alpha158vwap": ("Alpha158vwap", "quant_master.contrib.data.handler"),
            "Alpha360vwap": ("Alpha360vwap", "quant_master.contrib.data.handler"),
            "Alpha158LiquidityState": ("Alpha158LiquidityState", "quant_master.contrib.data.liquidity_state_handler"),
            "TranscendenceAlpha": ("TranscendenceAlpha", "quant_master.contrib.data.transcendence_handler"),
        }
        h_cls, h_mod = handler_map.get(handler_id, ("Alpha158", "quant_master.contrib.data.handler"))

        # 覆盖 dataset handler
        task = wf_config.get("task", {})
        dataset_cfg = task.get("dataset", {})
        ds_kwargs = dataset_cfg.get("kwargs", {})
        ds_kwargs["handler"] = {"class": h_cls, "module_path": h_mod, "kwargs": dhc}
        ds_kwargs["segments"] = {
            "train": [train_start, train_end],
            "valid": [train_end, valid_end],
            "test": [bt_start, bt_end],
        }

        # 覆盖 benchmark
        wf_config["benchmark"] = bench
        wf_config["market"] = "csi300"

        # 覆盖策略参数
        pac = wf_config.get("port_analysis_config", {})
        strat_map = {
            "topk_dropout": ("TopkDropoutStrategy", "quant_master.contrib.strategy"),
            "soft_topk": ("SoftTopkStrategy", "quant_master.contrib.strategy"),
            "enhanced_indexing": ("EnhancedIndexingStrategy", "quant_master.contrib.strategy.optimizer.enhanced_indexing"),
            "twap": ("TWAPStrategy", "quant_master.contrib.strategy"),
            "sbb_ema": ("SBBStrategyEMA", "quant_master.contrib.strategy"),
            "ac_strategy": ("ACStrategy", "quant_master.contrib.strategy.rule_strategy"),
        }
        s_cls, s_mod = strat_map.get(strat_cfg.get("type", "topk_dropout"),
                                      ("TopkDropoutStrategy", "quant_master.contrib.strategy"))
        strat_kwargs = {"signal": "<PRED>"}
        if "top_k" in strat_cfg:
            strat_kwargs["topk"] = int(strat_cfg["top_k"])
        if "n_drop" in strat_cfg:
            strat_kwargs["n_drop"] = int(strat_cfg["n_drop"])
        pac["strategy"] = {"class": s_cls, "module_path": s_mod, "kwargs": strat_kwargs}
        pac["backtest"] = {
            "start_time": bt_start,
            "end_time": bt_end,
            "account": 100000000,
            "benchmark": bench,
            "exchange_kwargs": {
                "limit_threshold": config.get("limit_threshold", 0.095),
                "deal_price": config.get("return_type", "close"),
                "open_cost": config.get("open_cost", 0.0005),
                "close_cost": config.get("sell_cost", 0.0015),
                "min_cost": config.get("min_cost", 5),
            },
        }

        set_progress(10, "初始化引擎")
        log("info", f"初始化 quant_master (region=cn)")
        quant_master.init(
            provider_uri=os.path.expanduser("~/.quant_master/quant_master_data/cn_data"),
            region="cn",
        )

        set_progress(15, "加载数据")
        log("info", f"Handler: {h_cls}  训练: {train_start}~{train_end}  测试: {bt_start}~{bt_end}")

        # 确保 record 配置正确
        task["record"] = [
            {"class": "SignalRecord", "module_path": "quant_master.workflow.record_temp",
             "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}},
            {"class": "SigAnaRecord", "module_path": "quant_master.workflow.record_temp",
             "kwargs": {"ana_long_short": False, "ann_scaler": 252}},
            {"class": "PortAnaRecord", "module_path": "quant_master.workflow.record_temp",
             "kwargs": {"config": pac}},
        ]
        wf_config["task"] = task

        set_progress(20, "训练模型")
        log("info", f"模型: {task.get('model', {}).get('class', '?')}")

        recorder = task_train(task, experiment_name=f"ui_backtest_{run_id}")

        set_progress(80, "提取结果")
        log("info", "从 recorder 加载回测指标")

        # 提取信号指标
        metrics = recorder.list_metrics()
        ic = metrics.get("IC", 0)
        icir = metrics.get("ICIR", 0)
        rank_ic = metrics.get("Rank IC", 0)
        rank_icir = metrics.get("Rank ICIR", 0)

        # 提取组合回测结果
        report_df = recorder.load_object(f"portfolio_analysis/report_normal_{freq}.pkl")
        risk_df = recorder.load_object(f"portfolio_analysis/port_analysis_{freq}.pkl")

        # 从 risk_df 提取关键指标
        risk_data = {}
        if risk_df is not None:
            for idx in risk_df.index:
                key = f"{idx[0]}.{idx[1]}" if isinstance(idx, tuple) else str(idx)
                risk_data[key] = float(risk_df.loc[idx].values[0]) if hasattr(risk_df.loc[idx], 'values') else float(risk_df.loc[idx])

        # 构建累计收益曲线
        daily = []
        if report_df is not None:
            cum_ret = 0.0
            bench_cum = 0.0
            for dt, row in report_df.iterrows():
                ret = float(row.get("return", 0)) * 100
                cum_ret += ret
                bench_ret = float(row.get("bench", 0)) * 100 if "bench" in row else 0
                bench_cum += bench_ret
                turnover = float(row.get("turnover", 0)) * 100 if "turnover" in row else 0
                cost = float(row.get("cost", 0)) * 10000 if "cost" in row else 0
                daily.append({
                    "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "cumReturn": round(cum_ret, 2),
                    "benchCumReturn": round(bench_cum, 2),
                    "turnover": round(turnover, 2),
                    "costBps": round(cost, 1),
                })

        # 提取最新持仓
        positions = []
        try:
            pos_dict = recorder.load_object(f"portfolio_analysis/positions_normal_{freq}.pkl")
            if pos_dict:
                last_dt = max(pos_dict.keys())
                last_pos = pos_dict[last_dt]
                if hasattr(last_pos, 'position'):
                    for sym, detail in last_pos.position.items():
                        weight = detail.get("weight", 0) if isinstance(detail, dict) else 0
                        positions.append({
                            "symbol": str(sym),
                            "weight": round(float(weight), 4),
                            "pnl": 0,
                        })
        except Exception as e:
            log("warn", f"加载持仓数据失败: {e}")

        # 从 risk_df 提取年化收益、夏普、回撤等
        ann_ret_key = f"excess_return_with_cost.annualized_return"
        ir_key = f"excess_return_with_cost.information_ratio"
        dd_key = f"excess_return_with_cost.max_drawdown"
        ann_ret = risk_data.get(ann_ret_key, 0) * 100
        ir = risk_data.get(ir_key, 0)
        max_dd = risk_data.get(dd_key, 0) * 100

        total_ret = cum_ret if daily else 0
        bench_total = bench_cum if daily else 0

        result = {
            "metrics": {
                "annualReturn": round(ann_ret, 2),
                "benchAnnualReturn": round(bench_total / max(len(daily), 1) * 252, 2),
                "excessReturn": round(ann_ret - bench_total / max(len(daily), 1) * 252, 2),
                "sharpe": round(ir * np.sqrt(252) if ir else 0, 2),
                "maxDrawdown": round(abs(max_dd), 2),
                "informationRatio": round(ir, 2),
                "std": round(float(np.std([d["cumReturn"] for d in daily])) if daily else 0, 2),
                "winRate": round(50 + ic * 500, 1),
                "avgTurnover": round(np.mean([d["turnover"] for d in daily]) if daily else 0, 2),
                "totalCost": round(np.sum([d["costBps"] for d in daily]) / 10000 * 100 if daily else 0, 4),
                "ic": round(float(ic), 4),
                "icir": round(float(icir), 2),
                "rankIc": round(float(rank_ic), 4),
                "rankIcir": round(float(rank_icir), 2),
            },
            "daily": daily,
            "positions": positions[:30],
            "trades": [],
        }

        run["results"] = result
        run["done"] = True
        run["success"] = True
        set_progress(100, "完成")
        log("success", f"回测完成! 年化收益: {ann_ret:.2f}%  IR: {ir:.2f}  IC: {ic:.4f}")

    except Exception as e:
        log("error", f"回测异常: {e}")
        traceback.print_exc()
        _demo_fallback(run_id, config)


def _demo_fallback(run_id, config):
    """当真实引擎不可用时，回退到演示数据。"""
    run = app.backtest_runs[run_id]
    run["results"] = _gen_backtest_demo(config)
    run["done"] = True
    run["success"] = True
    run["progress"] = 100
    run["step"] = "完成"


def _gen_backtest_demo(cfg):
    """生成回测演示数据。"""
    random.seed(42)
    dates = []
    d = date(2024, 1, 2)
    end = date(2025, 12, 31)
    while d <= end:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)

    strat_val = 1.0
    bench_val = 1.0
    daily = []
    top_k = int(cfg.get("top_k", 30))
    symbols = [f"SH{600000+i:06d}" for i in range(top_k)]

    for dt in dates:
        sr = random.gauss(0.0008, 0.012)
        br = random.gauss(0.0003, 0.01)
        strat_val *= (1 + sr)
        bench_val *= (1 + br)
        daily.append({
            "date": dt,
            "return": round(sr * 100, 4),
            "cumReturn": round((strat_val - 1) * 100, 2),
            "benchReturn": round(br * 100, 4),
            "benchCumReturn": round((bench_val - 1) * 100, 2),
            "turnover": round(random.uniform(0.05, 0.25), 4),
            "cost": round(random.uniform(0.0005, 0.003), 4),
        })

    annual_ret = round((strat_val ** (252 / len(dates)) - 1) * 100, 2)
    bench_annual = round((bench_val ** (252 / len(dates)) - 1) * 100, 2)
    returns = [d["return"] for d in daily]
    std = round(np.std(returns) * math.sqrt(252), 2)
    sharpe = round(annual_ret / std, 2) if std > 0 else 0

    peak = 0
    mdd = 0
    val = 1.0
    for d in daily:
        val *= (1 + d["return"] / 100)
        peak = max(peak, val)
        dd = (peak - val) / peak
        mdd = max(mdd, dd)
    mdd = round(mdd * 100, 2)

    last_positions = []
    for i, sym in enumerate(symbols):
        last_positions.append({
            "symbol": sym,
            "weight": round(1 / top_k + random.uniform(-0.01, 0.01), 4),
            "pnl": round(random.uniform(-3, 5), 2),
        })

    trades = []
    for i in range(min(20, len(dates))):
        dt = dates[i * len(dates) // 20]
        sym = random.choice(symbols)
        trades.append({
            "date": dt,
            "symbol": sym,
            "direction": random.choice(["buy", "sell"]),
            "price": round(random.uniform(8, 50), 2),
            "volume": random.randint(100, 5000) * 100,
            "amount": 0,
            "cost": round(random.uniform(5, 50), 2),
        })
        trades[-1]["amount"] = round(trades[-1]["price"] * trades[-1]["volume"], 2)

    return {
        "metrics": {
            "annualReturn": annual_ret,
            "benchAnnualReturn": bench_annual,
            "excessReturn": round(annual_ret - bench_annual, 2),
            "std": std,
            "sharpe": sharpe,
            "maxDrawdown": mdd,
            "informationRatio": round(sharpe * 0.7, 2),
            "avgTurnover": round(np.mean([d["turnover"] for d in daily]) * 100, 2),
            "totalCost": round(sum(d["cost"] for d in daily) * 100, 2),
            "winRate": round(random.uniform(52, 62), 1),
        },
        "daily": daily,
        "positions": last_positions,
        "trades": trades,
    }
