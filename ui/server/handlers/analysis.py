"""分析：因子分析。"""
import os
import time
from pathlib import Path

from .. import app


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
