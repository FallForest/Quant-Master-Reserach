"""模型选股：注册表加载 + 后台训练/预测任务。"""
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import yaml

_MODELS_FILE = Path(__file__).resolve().parent.parent / "models.yaml"

# 运行状态
_runs = {}
_counter = 0
_qm_lock = threading.Lock()


def load_registry():
    """从 models.yaml 加载模型和处理器注册表。"""
    with open(_MODELS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {}), data.get("handlers", {})


_MODEL_REGISTRY, _HANDLER_REGISTRY = load_registry()


def get_model_registry():
    return _MODEL_REGISTRY


def get_handler_registry():
    return _HANDLER_REGISTRY


def get_runs():
    return _runs


def get_counter():
    global _counter
    _counter += 1
    return _counter


def run_stock_selection(run_id, config, data_obj=None):
    """后台选股任务：初始化环境 → 构建数据集 → 训练模型 → 预测 → 返回 Top N。"""
    # 确保 quant_master 在 path 中
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    run = _runs.get(run_id)
    if not run:
        return

    def _log(level, msg):
        ts = time.strftime("%H:%M:%S")
        run["logs"].append({"level": level, "msg": f"[{ts}] {msg}"})

    def _set_progress(pct, step):
        run["progress"] = pct
        run["step"] = step

    try:
        # 1. 初始化
        _set_progress(5, "初始化环境")
        _log("info", "初始化 quant_master ...")
        provider_uri = os.path.expanduser("~/.quant_master/quant_master_data/tdx_cn_data")
        if not Path(provider_uri).exists():
            provider_uri = os.path.expanduser("~/.quant_master/quant_master_data/cn_data")
        with _qm_lock:
            import quant_master as qm
            qm.init(provider_uri=provider_uri, region="cn")

        # 2. 读取活跃股票
        _set_progress(10, "读取股票列表")
        test_date = config.get("test_date", "")
        universe = config.get("universe", "500")
        inst_path = Path(provider_uri) / "instruments" / "all.txt"
        stocks = []
        with open(inst_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] >= test_date:
                    stocks.append(parts[0])
        if not stocks:
            raise ValueError(f"未找到目标日期 {test_date} 的活跃股票")
        if universe != "all":
            limit = int(universe)
            stocks = stocks[:limit]
        _log("info", f"股票池: {len(stocks)} 只 (universe={universe})")

        # 3. 构建数据集
        _set_progress(15, "构建数据集")
        model_id = config.get("model_id", "lgb")
        handler_id = config.get("handler_id", "alpha158")
        model_spec = _MODEL_REGISTRY[model_id]
        handler_spec = _HANDLER_REGISTRY[handler_id]

        handler_mod = importlib.import_module(handler_spec["import_path"])
        HandlerCls = getattr(handler_mod, handler_spec["class_name"])

        train_start = config.get("train_start", "2018-01-01")
        train_end = config.get("train_end", "2023-12-31")
        valid_start = config.get("valid_start", "2024-01-01")
        valid_end = config.get("valid_end", "2025-06-30")

        _log("info", f"创建数据处理器 {handler_spec['label']} (共 {len(stocks)} 只股票) ...")
        _set_progress(16, "创建数据处理器")
        t_handler = time.time()
        from quant_master.data.dataset.processor import Fillna
        handler = HandlerCls(
            start_time=train_start,
            end_time=test_date,
            fit_start_time=train_start,
            fit_end_time=train_end,
            instruments=stocks,
            infer_processors=[Fillna()],
        )
        _log("info", f"数据处理器创建完成 ({time.time() - t_handler:.1f}s)")

        from quant_master.data.dataset import DatasetH
        _set_progress(20, "加载数据集")
        _log("info", "加载 DatasetH ...")
        t_ds = time.time()
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": [train_start, train_end],
                "valid": [valid_start, valid_end],
                "test": [test_date, test_date],
            },
        )
        _log("info", f"数据集加载完成 ({time.time() - t_ds:.1f}s)")
        _log("info", f"数据集: train={train_start}~{train_end}, valid={valid_start}~{valid_end}, test={test_date}")

        # 4. 加载模型
        _set_progress(25, "加载模型")
        model_mod = importlib.import_module(model_spec["import_path"])
        ModelCls = getattr(model_mod, model_spec["class_name"])
        params = model_spec["default_params"].copy()
        model = ModelCls(**params)
        _log("info", f"模型: {model_spec['label']} ({handler_spec['label']})")

        # 5. 训练 — fit() 是阻塞调用，用 bumper 线程递增进度
        _set_progress(30, "训练模型")
        _log("info", "开始训练 ...")
        stop_event = threading.Event()

        def _progress_bumper():
            pct = 30
            while not stop_event.is_set():
                pct = min(pct + 1, 89)
                run["progress"] = pct
                run["step"] = f"训练模型 ({pct}%)"
                time.sleep(2)

        bumper = threading.Thread(target=_progress_bumper, daemon=True)
        bumper.start()

        t0 = time.time()
        model.fit(dataset)
        elapsed_fit = time.time() - t0
        stop_event.set()

        _set_progress(90, "预测中")
        _log("info", f"训练完成 ({elapsed_fit:.1f}s)，开始预测 ...")

        # 6. 预测
        pred = model.predict(dataset, segment="test")

        # 7. 整理结果
        _set_progress(95, "整理结果")
        import pandas as pd
        if isinstance(pred.index, pd.MultiIndex):
            pred_df = pred.reset_index()
            pred_df.columns = ["datetime", "instrument", "score"]
            target_preds = pred_df[pred_df["datetime"] == test_date].copy()
        else:
            target_preds = pd.DataFrame({"instrument": pred.index, "score": pred.values})

        if target_preds.empty:
            raise ValueError("预测结果为空")

        target_preds = target_preds.sort_values("score", ascending=False).reset_index(drop=True)

        names = data_obj.get_names() if data_obj else {}

        top_n = int(config.get("top_n", 50))
        results = []
        for i, row in target_preds.head(top_n).iterrows():
            sym = str(row["instrument"])
            code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
            results.append({
                "rank": i + 1,
                "symbol": sym,
                "name": names.get(code6, ""),
                "score": round(float(row["score"]), 6),
            })

        bottom = []
        for i, row in target_preds.tail(10).iterrows():
            sym = str(row["instrument"])
            code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
            bottom.append({
                "rank": int(i) + 1,
                "symbol": sym,
                "name": names.get(code6, ""),
                "score": round(float(row["score"]), 6),
            })

        _set_progress(100, "完成")
        _log("success", f"选股完成! Top {top_n} 已生成，共 {len(target_preds)} 只股票")

        run["results"] = {"top": results, "bottom": bottom, "total": len(target_preds)}
        run["done"] = True
        run["success"] = True

        try:
            out_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"stock_select_{run_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "config": config,
                    "results": run["results"],
                    "timestamp": time.time(),
                }, f, ensure_ascii=False, indent=2)
            _log("info", f"结果已保存: {out_path}")
        except Exception as e:
            _log("warn", f"持久化失败: {e}")

    except Exception as e:
        _log("error", f"选股失败: {e}")
        run["done"] = True
        run["success"] = False
        run["error"] = str(e)
        _set_progress(run["progress"], f"失败: {e}")
