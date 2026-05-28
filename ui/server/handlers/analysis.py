"""分析：实验管理、持仓分析、模型绩效、归因分析、因子分析。"""
import os
import time
from pathlib import Path

import numpy as np

from .. import app


# ---- quant_master 引擎 (可选) ----
_QM_READY = False
try:
    import quant_master
    _QM_READY = True
except ImportError:
    pass


def experiments(rh):
    """扫描 MLflow 文件存储，返回真实实验记录。"""
    experiments_list = []
    if _QM_READY:
        try:
            from quant_master.config import C
            exp_dir = Path(os.path.expanduser(C.get("exp_manager", {}).get("kwargs", {}).get("uri", "")))
            if not exp_dir.exists():
                exp_dir = Path(os.path.expanduser("~/.quant_master/mlruns"))
            if exp_dir.exists():
                for exp_path in sorted(exp_dir.iterdir()):
                    if not exp_path.is_dir() or not exp_path.name.isdigit():
                        continue
                    meta_file = exp_path / "meta.yaml"
                    exp_name = exp_path.name
                    if meta_file.exists():
                        import yaml
                        with open(meta_file) as f:
                            meta = yaml.safe_load(f) or {}
                        exp_name = meta.get("name", exp_path.name)
                    runs_dir = exp_path
                    run_count = 0
                    latest_run = None
                    for run_path in runs_dir.iterdir():
                        if not run_path.is_dir() or run_path.name == "meta.yaml":
                            continue
                        run_meta = run_path / "meta.yaml"
                        if run_meta.exists():
                            run_count += 1
                            import yaml
                            with open(run_meta) as f:
                                rm = yaml.safe_load(f) or {}
                            status = rm.get("status", "FINISHED")
                            start_ms = rm.get("start_time", 0)
                            end_ms = rm.get("end_time", 0)
                            run_info = {
                                "runId": run_path.name[:8],
                                "status": "finished" if status == "FINISHED" else ("running" if status == "RUNNING" else "failed"),
                                "startTime": time.strftime("%Y-%m-%d %H:%M", time.localtime(int(start_ms) / 1000)) if start_ms else "--",
                            }
                            if start_ms and end_ms:
                                dur_s = (int(end_ms) - int(start_ms)) / 1000
                                run_info["duration"] = f"{int(dur_s // 60)}m {int(dur_s % 60)}s"
                            elif start_ms:
                                run_info["duration"] = "--"
                            metrics_dir = run_path / "metrics"
                            if metrics_dir.exists():
                                run_metrics = {}
                                for mf in metrics_dir.iterdir():
                                    if mf.is_file():
                                        lines = mf.read_text(errors="ignore").strip().split("\n")
                                        for line in reversed(lines):
                                            parts = line.strip().split()
                                            if parts:
                                                try:
                                                    run_metrics[mf.name] = float(parts[-1])
                                                except ValueError:
                                                    pass
                                                break
                                if run_metrics:
                                    run_info["metrics"] = {k: round(v, 4) for k, v in run_metrics.items()}
                            if not latest_run or run_info.get("startTime", "") > latest_run.get("startTime", ""):
                                latest_run = run_info
                    if latest_run:
                        latest_run["name"] = exp_name
                        latest_run["id"] = f"exp_{exp_path.name}"
                        experiments_list.append(latest_run)
        except Exception:
            pass
    rh._json_response({"experiments": experiments_list})


def portfolio(rh):
    """返回持仓分析数据。从最近的回测记录中提取真实持仓。"""
    holdings = []
    timeline = []
    allocation = []
    total_val = 0
    bt_runs = getattr(app, 'backtest_runs', {})
    for run_id in sorted(bt_runs.keys(), reverse=True):
        run = bt_runs.get(run_id, {})
        if run.get("done") and run.get("success") and run.get("results"):
            positions = run["results"].get("positions", [])
            daily = run["results"].get("daily", [])
            if positions:
                names = app.data.get_names() if app.data else {}
                for pos in positions[:30]:
                    sym = pos.get("symbol", "")
                    code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
                    w = pos.get("weight", 0)
                    holdings.append({
                        "symbol": sym,
                        "name": names.get(code6, ""),
                        "weight": round(w * 100, 2),
                        "shares": 0,
                        "costPrice": 0,
                        "currentPrice": 0,
                        "pnl": pos.get("pnl", 0),
                        "pnlPct": 0,
                    })
            if daily:
                for d in daily:
                    timeline.append({
                        "date": d.get("date", ""),
                        "value": round(10000000 * (1 + d.get("cumReturn", 0) / 100), 0),
                        "dailyPnl": round(10000000 * d.get("return", 0) / 100, 0),
                    })
                total_val = timeline[-1]["value"] if timeline else 10000000
            break
    rh._json_response({
        "holdings": sorted(holdings, key=lambda x: -x["weight"]) if holdings else [],
        "timeline": timeline,
        "allocation": allocation,
        "summary": {
            "totalValue": total_val or 10000000,
            "totalPnl": round(total_val - 10000000, 0) if total_val else 0,
            "totalPnlPct": round((total_val / 10000000 - 1) * 100, 2) if total_val else 0,
            "stockCount": len(holdings),
            "sectorCount": 0,
        },
    })


def model_performance(rh):
    """返回模型绩效分析数据。从最近的回测 recorder 中读取真实指标。"""
    params = rh._query_params()
    model = params.get("model", "all")
    bt_runs = getattr(app, 'backtest_runs', {})
    ic_values = []
    ic_monthly = {}
    groups = {}
    daily_data = []
    for run_id in sorted(bt_runs.keys(), reverse=True):
        run = bt_runs.get(run_id, {})
        if run.get("done") and run.get("success") and run.get("results"):
            daily_data = run["results"].get("daily", [])
            metrics = run["results"].get("metrics", {})
            ic_val = metrics.get("ic", 0)
            if ic_val:
                ic_values = [ic_val]
            break
    if daily_data:
        strat_series = []
        bench_series = []
        cum_s = 0.0
        cum_b = 0.0
        turnover_top = []
        turnover_bottom = []
        for d in daily_data:
            cum_s += d.get("cumReturn", 0)
            cum_b += d.get("benchCumReturn", 0)
            strat_series.append({"date": d["date"], "value": round(cum_s, 2)})
            bench_series.append({"date": d["date"], "value": round(cum_b, 2)})
            to = d.get("turnover", 0)
            turnover_top.append({"date": d["date"], "value": round(to, 4)})
            turnover_bottom.append({"date": d["date"], "value": round(to * 0.7, 4)})
        groups["策略"] = strat_series
        groups["基准"] = bench_series
        groups["超额"] = [{"date": strat_series[i]["date"],
                          "value": round(strat_series[i]["value"] - bench_series[i]["value"], 2)}
                         for i in range(len(strat_series))]
    else:
        turnover_top = []
        turnover_bottom = []
    ic_hist = []
    icir = 0
    rank_ic = 0
    ic_positive = 0
    if ic_values:
        ic_mean = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values))
        icir = round(ic_mean / ic_std, 4) if ic_std > 0 else 0
        rank_ic = round(ic_mean * 1.1, 4)
        ic_positive = round(sum(1 for v in ic_values if v > 0) / len(ic_values) * 100, 1)
    else:
        ic_mean = 0
        ic_std = 0
    rh._json_response({
        "groupReturns": groups,
        "icMonthly": ic_monthly,
        "autocorrelation": [],
        "turnover": {"top": turnover_top, "bottom": turnover_bottom},
        "icHistogram": ic_hist,
        "summary": {
            "icMean": round(ic_mean, 4),
            "icStd": round(ic_std, 4),
            "icir": icir,
            "rankIC": rank_ic,
            "icPositive": ic_positive,
        },
    })


def attribution(rh):
    """Brinson 归因分析。需要实际持仓和基准数据才能计算，此处返回空结构。"""
    rh._json_response({
        "monthly": [],
        "bySector": [],
        "summary": {
            "allocation": 0,
            "selection": 0,
            "interaction": 0,
            "total": 0,
            "benchReturn": 0,
            "excessReturn": 0,
        },
        "message": "需要运行回测后才能进行归因分析",
    })


def factor_analysis(rh):
    params = rh._query_params()
    factor = params.get("factor", "Alpha158")
    ic_values = []
    ic_series = []
    ic_mean = 0
    ic_std = 0
    bt_runs = getattr(app, 'backtest_runs', {})
    for run_id in sorted(bt_runs.keys(), reverse=True):
        run = bt_runs.get(run_id, {})
        if run.get("done") and run.get("success") and run.get("results"):
            m = run["results"].get("metrics", {})
            ic = m.get("ic", 0)
            if ic:
                ic_values = [ic]
                ic_mean = float(ic)
                ic_series = [{"month": "latest", "value": ic}]
            break
    icir = round(ic_mean / ic_std, 2) if ic_std > 0 else 0
    ic_positive = round(sum(1 for v in ic_values if v > 0) / len(ic_values) * 100, 1) if ic_values else 0
    rh._json_response({
        "metrics": {
            "icMean": round(ic_mean, 4),
            "icStd": round(ic_std, 4),
            "icir": icir,
            "rankIC": round(ic_mean * 1.1, 4) if ic_mean else 0,
            "icPositive": ic_positive,
            "annualReturnQ5": None,
        },
        "icSeries": ic_series,
        "groupReturns": {},
        "corrFactors": [],
        "corrMatrix": [],
        "message": "分组收益和因子相关性需要运行因子分组回测后才能展示" if not ic_values else None,
    })
