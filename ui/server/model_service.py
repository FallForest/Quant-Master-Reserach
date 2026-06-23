"""模型服务层：从 MLflow 加载实验记录，提供选股查询接口。"""
import bisect
import json
import logging
import pickle
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import mlflow.tracking
import pandas as pd

from quant_master.data import D
from quant_master.data.cache import H
from quant_master.data.data import ProviderBackendMixin
from quant_master.data.dataset import DatasetH
from quant_master.data.dataset.handler import DataHandlerLP
from quant_master.utils import lazy_sort_index

from . import app
from .calendar_validation import InvalidCalendarError
from .datadir import describe_trading_day, get_effective_data_dir, get_trading_calendar
from .sync import get_data_health_snapshot, get_sync_status

logger = logging.getLogger(__name__)
# 确保 logger 有 handler 输出到控制台（main.py 通过 uvicorn 启动时不会设 basicConfig）
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

# 项目根目录下的 mlruns
from .config import LIVE_DATA_DIR, MLRUNS_URI, PROJECT_ROOT
_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
_PRED_LOCAL_DIR = Path(__file__).parent / "pred_cache"
_INDEX_SIZE_LIMITS = {
    "csi100": (80, 130),
    "csi300": (250, 350),
    "csi500": (430, 570),
    "csi800": (700, 900),
    "csi1000": (900, 1100),
}
_INSTRUMENT_PARAM_KEYS = (
    "dataset.kwargs.handler.kwargs.instruments",
    "task.dataset.kwargs.handler.kwargs.instruments",
    "instruments",
)


def drop_invalid_live_predictions(df, market):
    """Drop live prediction groups whose daily size violates the declared index universe."""
    limits = _INDEX_SIZE_LIMITS.get(str(market).lower()) if market else None
    if not limits or not isinstance(df, pd.DataFrame) or "source" not in df.columns:
        return df, 0
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        return df, 0

    lower, upper = limits
    date_index = df.index.get_level_values(0)
    live_mask = df["source"].eq("live")
    if not live_mask.any():
        return df, 0

    live_counts = df.loc[live_mask].groupby(date_index[live_mask]).size()
    bad_dates = live_counts[(live_counts < lower) | (live_counts > upper)].index
    if len(bad_dates) == 0:
        return df, 0

    bad_mask = live_mask & date_index.isin(bad_dates)
    return df.loc[~bad_mask].copy(), int(bad_mask.sum())


class ModelService:
    """封装 MLflow 数据访问，支持按 alias 注册多个模型。"""

    @staticmethod
    @contextmanager
    def _feature_only_loader(handler):
        loader = getattr(handler, "data_loader", None)
        fields = getattr(loader, "fields", None)
        if not getattr(loader, "is_group", False) or not isinstance(fields, dict) or "feature" not in fields:
            yield False
            return

        original_fields = fields
        loader.fields = {"feature": original_fields["feature"]}
        try:
            yield True
        finally:
            loader.fields = original_fields

    @staticmethod
    def _rebuild_feature_only_infer(handler, start_time, end_time):
        handler._data = lazy_sort_index(handler.data_loader.load(handler.instruments, start_time, end_time))

        shared_df = handler._data
        if not handler._is_proc_readonly(handler.shared_processors):
            shared_df = shared_df.copy()
        shared_df = handler._run_proc_l(shared_df, handler.shared_processors, with_fit=False, check_for_infer=True)

        infer_df = shared_df
        if not handler._is_proc_readonly(handler.infer_processors):
            infer_df = infer_df.copy()
        infer_df = handler._run_proc_l(infer_df, handler.infer_processors, with_fit=False, check_for_infer=True)
        handler._infer = infer_df

    def __init__(self, registry_path=None):
        self._registry_path = Path(registry_path) if registry_path else _REGISTRY_PATH
        self._registry = self._load_registry()
        self._client = mlflow.tracking.MlflowClient(tracking_uri=MLRUNS_URI)
        self.data = app.data
        self._pred_cache = {}   # alias → DataFrame (历史预测)
        self._info_cache = {}   # alias → dict
        self._model_cache = {}  # alias → fitted Model object
        self._dataset_cache = {}  # alias → DatasetH object
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _calendar_error_message(exc: InvalidCalendarError):
        return exc.to_user_message()

    @staticmethod
    def _next_weekday(date_value):
        candidate = pd.Timestamp(date_value) + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate



    def _validate_live_prediction_timing(self, requested_date, feature_date, trading_dates, snapshot):
        requested_ts = pd.Timestamp(requested_date)
        feature_ts = pd.Timestamp(feature_date)
        sync_status = get_sync_status()
        requested_str = requested_ts.strftime("%Y-%m-%d")
        feature_str = feature_ts.strftime("%Y-%m-%d")

        if requested_ts <= feature_ts:
            return {
                "allowed": True,
                "mode": "same_day",
                "message": "",
                "syncing": bool(sync_status.get("running")),
            }

        # requested > feature：校验是否紧挨 feature 的下一个交易日
        feature_info = describe_trading_day(feature_str, trading_dates)
        expected_next = feature_info.get("next")
        if not expected_next:
            expected_next = self._next_weekday(feature_ts).strftime("%Y-%m-%d")
        if requested_str != expected_next:
            raise ValueError(
                f"目标交易日 {requested_str} 不是数据日 {feature_str} 的下一个交易日，"
                "当前仅支持收盘后生成下一交易日选股。"
            )

        # 数据是否足够覆盖 feature_date（替代硬性时间门禁）
        market_effective = snapshot.get("marketEffectiveLastDate")
        if not market_effective or pd.Timestamp(market_effective) < feature_ts:
            if sync_status.get("running"):
                raise ValueError("数据同步正在进行中，请等待同步完成后再运行下一交易日选股。")
            raise ValueError(
                f"最新市场有效数据尚未推进到 {feature_str}，请先完成当日收盘数据同步后再运行下一交易日选股。"
            )

        # 同步标记卡死但数据已到位，允许通过
        if sync_status.get("running"):
            logger.warning(
                "Sync is marked running but data is already fresh (%s >= %s), "
                "allowing prediction to proceed.",
                market_effective, feature_str,
            )

        return {
            "allowed": True,
            "mode": "next_trading_day",
            "message": f"目标交易日 {requested_str} 将使用 {feature_str} 收盘数据生成。",
            "syncing": False,
        }

    def _load_registry(self):
        if not self._registry_path.exists():
            logger.warning("Model registry not found: %s", self._registry_path)
            return {}
        with open(self._registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload_registry(self):
        """热加载注册表（用于运行时动态添加模型）。"""
        with self._lock:
            self._registry = self._load_registry()
            self._pred_cache.clear()
            self._info_cache.clear()
            self._model_cache.clear()
            self._dataset_cache.clear()

    def _get_run(self, alias):
        entry = self._registry.get(alias)
        if not entry:
            raise KeyError(f"Model alias '{alias}' not found in registry")
        run = self._client.get_run(entry["run_id"])
        return run, entry

    def _load_predictions(self, alias):
        """加载 pred.pkl 并缓存 DataFrame。优先读取本地追加缓存。"""
        with self._lock:
            if alias in self._pred_cache:
                return self._pred_cache[alias]

        # 优先读取本地追加后的缓存
        local_path = _PRED_LOCAL_DIR / f"{alias}_pred.pkl"
        if local_path.is_file():
            df = pd.read_pickle(local_path)
            if isinstance(df.index, pd.MultiIndex):
                market = self._declared_universe(alias)
                df, dropped_rows = drop_invalid_live_predictions(df, market)
                if dropped_rows:
                    logger.warning(
                        "Dropped %d invalid live prediction rows for '%s' from local cache (market=%s)",
                        dropped_rows,
                        alias,
                        market,
                    )
                with self._lock:
                    self._pred_cache[alias] = df
                logger.info("Loaded pred.pkl from local cache for '%s': %d rows", alias, len(df))
                return df

        _, entry = self._get_run(alias)
        run_id = entry["run_id"]
        try:
            artifact_path = self._client.download_artifacts(run_id, "pred.pkl")
        except Exception as exc:
            raise FileNotFoundError(
                f"pred.pkl not found for model '{alias}' (run_id={run_id})"
            ) from exc

        df = pd.read_pickle(artifact_path)
        # 确保索引层级正确
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(f"pred.pkl for '{alias}' must have MultiIndex (datetime, instrument)")

        with self._lock:
            self._pred_cache[alias] = df
        logger.info("Loaded pred.pkl for '%s': %d rows", alias, len(df))
        return df

    def _load_artifact(self, run_id, name):
        """从 MLflow artifacts 下载并反序列化 pickle 对象。"""
        path = self._client.download_artifacts(run_id, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _prediction_cache_path(alias):
        return _PRED_LOCAL_DIR / f"{alias}_pred.pkl"

    @staticmethod
    def _normalize_instruments(values):
        return {str(value).upper() for value in values}

    @staticmethod
    def _validate_market_size(market, count, context):
        limits = _INDEX_SIZE_LIMITS.get(str(market).lower())
        if not limits:
            return
        lower, upper = limits
        if count < lower or count > upper:
            raise ValueError(
                f"Universe guard failed: {context} for {market} resolved to {count} instruments, "
                f"outside expected range [{lower}, {upper}]. Refusing to persist live prediction."
            )

    def _clear_runtime_universe_caches(self, alias=None):
        """Clear process caches that may hold stale instrument memberships."""
        try:
            H["i"].clear()
        except Exception:
            logger.exception("Failed to clear quant_master instrument cache")
        with self._lock:
            if alias:
                self._dataset_cache.pop(alias, None)
                self._pred_cache.pop(alias, None)
            else:
                self._dataset_cache.clear()
                self._pred_cache.clear()

    def _declared_universe(self, alias, entry=None, run=None):
        entry = entry or self._registry.get(alias, {})
        registry_market = None
        for key in ("instruments", "universe"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                registry_market = value.strip().lower()
                break

        if run is None:
            try:
                run, _ = self._get_run(alias)
            except Exception:
                return registry_market
        params = getattr(getattr(run, "data", None), "params", {}) or {}
        run_market = None
        for key in _INSTRUMENT_PARAM_KEYS:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                run_market = value.strip().lower()
                break

        if registry_market and run_market and registry_market != run_market:
            raise ValueError(
                f"Universe guard failed: model '{alias}' registry declares {registry_market}, "
                f"but MLflow run declares {run_market}."
            )
        return registry_market or run_market

    def _expected_universe_for_date(self, market, feature_date):
        market = str(market).lower()
        feature_date = pd.Timestamp(feature_date)
        H["i"].clear()
        instruments = D.instruments(market)
        expected = D.list_instruments(
            instruments,
            start_time=feature_date,
            end_time=feature_date,
            freq="day",
            as_list=True,
        )
        expected_set = self._normalize_instruments(expected)
        self._validate_market_size(market, len(expected_set), f"data provider universe on {feature_date:%Y-%m-%d}")
        return expected_set

    def _validate_live_prediction_universe(self, alias, day_data, feature_date):
        run, entry = self._get_run(alias)
        market = self._declared_universe(alias, entry=entry, run=run)
        if not market or market == "all":
            return None

        actual_set = self._normalize_instruments(day_data.index)
        self._validate_market_size(market, len(actual_set), f"prediction result on {pd.Timestamp(feature_date):%Y-%m-%d}")

        try:
            expected_set = self._expected_universe_for_date(market, feature_date)
        except ValueError:
            self._clear_runtime_universe_caches(alias)
            raise

        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            self._clear_runtime_universe_caches(alias)
            raise ValueError(
                f"Universe guard failed: model '{alias}' declares {market}, but prediction universe does not match "
                f"{pd.Timestamp(feature_date):%Y-%m-%d}. expected={len(expected_set)}, actual={len(actual_set)}, "
                f"missing={len(missing)}, extra={len(extra)}."
            )
        return {"market": market, "expected": len(expected_set), "actual": len(actual_set)}

    def _persist_live_predictions(self, alias, day_data, requested_date, feature_date, resolved):
        persist_date = pd.Timestamp(requested_date)
        persist_date_str = persist_date.strftime("%Y-%m-%d")
        requested_date_str = pd.Timestamp(requested_date).strftime("%Y-%m-%d")
        feature_date_str = pd.Timestamp(feature_date).strftime("%Y-%m-%d")

        persist_df = day_data.copy()
        if isinstance(persist_df, pd.Series):
            persist_df = persist_df.to_frame("score")
        if "score" not in persist_df.columns:
            raise ValueError("Prediction result must contain a 'score' column")

        persist_df.index = pd.Index([str(instrument).upper() for instrument in persist_df.index], name="instrument")
        persist_df.insert(0, "date", persist_date)
        persist_df = persist_df.reset_index().set_index(["date", "instrument"]).sort_index()
        persist_df["requested_date"] = requested_date_str
        persist_df["feature_date"] = feature_date_str
        persist_df["date_mapped"] = bool(resolved.get("dateMapped"))
        persist_df["source"] = "live"
        persist_df["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cache_path = self._prediction_cache_path(alias)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        history_df = self._load_predictions(alias).copy()
        if not isinstance(history_df.index, pd.MultiIndex):
            raise ValueError(f"Prediction cache for '{alias}' must have MultiIndex (datetime, instrument)")

        if "source" in history_df.columns:
            live_mask = history_df["source"].eq("live")
            same_feature_mask = history_df.index.get_level_values(0) == persist_date
            history_df = history_df.loc[~(live_mask & same_feature_mask)]
        merged_df = pd.concat([history_df, persist_df], axis=0, sort=False).sort_index()

        with self._lock:
            merged_df.to_pickle(cache_path)
            self._pred_cache[alias] = merged_df

        logger.info(
            "Persisted live predictions for '%s': requested=%s, feature=%s, rows=%d, path=%s",
            alias,
            requested_date_str,
            feature_date_str,
            len(persist_df),
            cache_path,
        )
        return merged_df

    @staticmethod
    def _prediction_metadata(row, selected_date):
        selected_date_str = selected_date.strftime("%Y-%m-%d")

        def _first_value(key, default=None):
            if key not in row.index:
                return default
            value = row.get(key)
            if isinstance(value, pd.Series):
                if value.empty:
                    return default
                value = value.iloc[0]
            return default if pd.isna(value) else value

        requested_date = _first_value("requested_date", selected_date_str)
        feature_date = _first_value("feature_date", selected_date_str)
        date_mapped = bool(_first_value("date_mapped", False))
        source = _first_value("source", "historical") or "historical"
        return {
            "requestedDate": str(requested_date),
            "featureDate": str(feature_date),
            "dateMapped": date_mapped,
            "source": str(source),
        }

    def _load_model(self, alias):
        """加载并缓存拟合好的模型对象。"""
        with self._lock:
            if alias in self._model_cache:
                return self._model_cache[alias]

        _, entry = self._get_run(alias)
        run_id = entry["run_id"]
        logger.info("Loading model for '%s' (run_id=%s)...", alias, run_id)
        model = self._load_artifact(run_id, "params.pkl")

        # 补全旧版 pickle 可能缺失的属性（向后兼容）
        _MODEL_DEFAULTS = {
            "robust_rank_blend": 0.0,
            "prediction_shrinkage": 1.0,
            "robust_rank_blend_grid": [0.0],
            "prediction_shrinkage_grid": [1.0],
            "_final_score_control_grid_opt_in": False,
        }
        for attr, default in _MODEL_DEFAULTS.items():
            if not hasattr(model, attr):
                setattr(model, attr, default)
                logger.info("Set missing model attr %s = %s (backward compat)", attr, default)

        with self._lock:
            self._model_cache[alias] = model
        logger.info("Model loaded for '%s': %s", alias, type(model).__name__)
        return model

    def _get_latest_available_day(self):
        calendar = D.calendar(freq="day")
        if len(calendar) == 0:
            raise ValueError("No available day calendar in provider data")
        return pd.Timestamp(calendar[-1])

    def _get_live_data_status(self):
        effective_dir = get_effective_data_dir(self.data)
        snapshot = get_data_health_snapshot(effective_dir)
        sync_status = get_sync_status()
        try:
            trading_dates = get_trading_calendar(self.data, freq="day")
        except InvalidCalendarError as exc:
            raise ValueError(self._calendar_error_message(exc)) from exc
        latest_calendar_date = snapshot.get("calendarLastDate") or (trading_dates[-1] if trading_dates else None)
        feature_date = (
            snapshot.get("marketEffectiveLastDate")
            or snapshot.get("effectiveLastDate")
            or latest_calendar_date
        )
        return {
            "effectiveDir": effective_dir,
            "snapshot": snapshot,
            "syncStatus": sync_status,
            "tradingDates": trading_dates,
            "latestCalendarDate": latest_calendar_date,
            "featureDate": feature_date,
        }

    def _resolve_live_prediction_dates(self, requested_date: pd.Timestamp):
        status = self._get_live_data_status()
        feature_date = status["featureDate"]
        if not feature_date:
            raise ValueError("No effective trading data is available in the current data directory. Please sync data first.")

        feature_ts = pd.Timestamp(feature_date)
        requested_str = requested_date.strftime("%Y-%m-%d")
        trading_info = describe_trading_day(requested_str, status["tradingDates"])

        if trading_info.get("next") and requested_str < status["tradingDates"][0]:
            raise ValueError(trading_info["message"])

        mapped = requested_date > feature_ts
        if not mapped and trading_info.get("ok") is False and trading_info.get("previous"):
            feature_ts = pd.Timestamp(trading_info["previous"])
            mapped = feature_ts != requested_date

        if feature_ts > requested_date:
            raise ValueError(trading_info["message"])

        # 与原生 quant_master 语义一致：预测目标日 T 始终使用 T-1 的数据
        # 参见 quant_master/contrib/online/operator.py:64  get_pre_trading_date(trade_date)
        if not mapped and requested_date == feature_ts:
            prev_date = trading_info.get("previous")
            if prev_date is not None:
                prev_ts = pd.Timestamp(prev_date)
                if prev_ts < requested_date:
                    feature_ts = prev_ts
                    mapped = True
                    logger.info(
                        "Mapped same-day requested_date %s to previous trading day %s",
                        requested_str, prev_date,
                    )

        timing = self._validate_live_prediction_timing(
            requested_date=requested_date,
            feature_date=feature_ts,
            trading_dates=status["tradingDates"],
            snapshot=status["snapshot"],
        )
        message = timing.get("message", "")
        if mapped and not message:
            message = (
                f"目标交易日 {requested_str} 超出本地可用数据范围，"
                f"已使用最近可用数据日 {feature_ts.strftime('%Y-%m-%d')} 生成选股结果。"
            )

        return {
            "requestedDate": requested_date,
            "featureDate": feature_ts,
            "dateMapped": mapped,
            "message": message,
            "timingMode": timing.get("mode"),
            "syncing": timing.get("syncing", False),
            "latestCalendarDate": status["latestCalendarDate"],
            "marketEffectiveLastDate": status["snapshot"].get("marketEffectiveLastDate"),
            "effectiveLastDate": status["snapshot"].get("effectiveLastDate"),
        }

    def _ensure_handler_data(self, dataset, target_date=None):
        """确保 handler 在反序列化后拥有 _infer / _learn 等处理后数据。

        不走 setup_data(IT_LS) 路线——该模式会从 pickle 属性重新初始化 handler，
        覆盖手动修改。改为直接清缓存后调用 feature-only 加载。
        """
        handler = getattr(dataset, "handler", None)
        if handler is None:
            return

        needs_rebuild = not hasattr(handler, "_infer") or handler._infer is None

        if not needs_rebuild and target_date is not None:
            max_infer_date = handler._infer.index.get_level_values(0).max()
            if target_date > max_infer_date:
                needs_rebuild = True

        if not needs_rebuild:
            return

        # 清 feature storage 缓存，确保读到的文件长度是最新的
        with ProviderBackendMixin._storage_cache_lock:
            ProviderBackendMixin._storage_cache.clear()

        if not isinstance(dataset, DatasetH) or not isinstance(handler, DataHandlerLP):
            raise RuntimeError(
                f"无法为 {type(handler).__name__} 重建 _infer 数据："
                "仅支持 DatasetH/DataHandlerLP。"
            )

        start_time = getattr(handler, "start_time", None)
        end_time = getattr(handler, "end_time", None)

        # clamp end_time 到 calendar 范围，防止 pickle 中 2026-12-31 导致请求超出 calendar
        if target_date is not None and (end_time is None or target_date > pd.Timestamp(end_time)):
            end_time = target_date
        cal_list = D.calendar(freq="day")
        if end_time is not None and len(cal_list):
            cal_last = str(cal_list[-1])[:10]
            if pd.Timestamp(end_time) > pd.Timestamp(cal_last):
                end_time = cal_last

        # 推理阶段只需要足够计算 rolling 窗口的历史数据，不需要 2012 年起全量
        # Alpha158 最大滚动窗口 60 天，留余量用 90 天
        if target_date is not None and start_time is not None:
            min_start = (pd.Timestamp(target_date) - timedelta(days=90)).strftime("%Y-%m-%d")
            if pd.Timestamp(start_time) < pd.Timestamp(min_start):
                start_time = min_start

        handler.end_time = end_time
        if hasattr(handler, "data_loader") and hasattr(handler.data_loader, "end_time"):
            handler.data_loader.end_time = end_time

        # 直接 feature-only 重建（不经过 IT_LS 的父类重新初始化，避免覆盖手动修改）
        with self._feature_only_loader(handler):
            self._rebuild_feature_only_infer(handler, start_time, end_time)

        rebuilt_infer = getattr(handler, "_infer", None)
        if rebuilt_infer is None:
            raise RuntimeError(
                f"无法为 {type(handler).__name__} 重建 _infer 数据。"
            )

        # 日期截断检查——数据加载成功但最大日期不够：让 _predict_or_fallback 处理
        if target_date is not None:
            max_infer_date = rebuilt_infer.index.get_level_values(0).max()
            if pd.Timestamp(target_date) > max_infer_date:
                logger.warning(
                    "Feature-only rebuild truncated: max_infer_date=%s < target=%s. "
                    "Prediction may fall back to earlier date.",
                    max_infer_date, target_date,
                )

    def _load_dataset(self, alias, target_date=None):
        """加载并缓存 DatasetH 对象。"""
        with self._lock:
            if alias in self._dataset_cache:
                return self._dataset_cache[alias]

        _, entry = self._get_run(alias)
        run_id = entry["run_id"]
        logger.info("Loading dataset for '%s' (run_id=%s)...", alias, run_id)
        dataset = self._load_artifact(run_id, "dataset")

        # 确保 handler 有处理后的数据（_infer/_learn）
        self._ensure_handler_data(dataset, target_date=target_date)

        with self._lock:
            self._dataset_cache[alias] = dataset
        logger.info("Dataset loaded for '%s': %s", alias, type(dataset).__name__)
        return dataset

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def list_models(self):
        """返回已注册模型列表。"""
        result = []
        for alias, entry in self._registry.items():
            info = {k: v for k, v in entry.items() if k != "run_id"}
            info["alias"] = alias
            result.append(info)
        return result

    def get_model_info(self, alias):
        """返回模型元信息 + 指标 + 参数。"""
        with self._lock:
            if alias in self._info_cache:
                return self._info_cache[alias]

        run, entry = self._get_run(alias)
        data = run.data
        metrics = data.metrics
        params = data.params

        info = {
            "alias": alias,
            "name": entry.get("name", alias),
            "description": entry.get("description", ""),
            "runId": entry["run_id"],
            "status": str(run.info.status),
            "startTime": _fmt_time(run.info.start_time),
            "endTime": _fmt_time(run.info.end_time),
            "metrics": _pick(metrics, [
                ("1day.excess_return_without_cost.annualized_return", "annualizedReturn"),
                ("1day.excess_return_without_cost.information_ratio", "informationRatio"),
                ("1day.excess_return_without_cost.max_drawdown", "maxDrawdown"),
                ("1day.excess_return_without_cost.mean", "meanDailyReturn"),
                ("1day.excess_return_without_cost.std", "stdDailyReturn"),
                ("1day.excess_return_with_cost.mean", "meanReturnWithCost"),
                ("1day.excess_return_with_cost.max_drawdown", "maxDrawdownWithCost"),
                ("ICIR", "icir"),
                ("Rank ICIR", "rankIcir"),
            ]),
            "params": _pick(params, [
                ("model.class", "modelClass"),
                ("dataset.handler", "handler"),
                ("instruments", "instruments"),
                ("topk", "topk"),
                ("benchmark", "benchmark"),
            ]),
            "dateRange": {
                "testStart": params.get("test_start", ""),
                "testEnd": params.get("test_end", ""),
            },
        }

        with self._lock:
            self._info_cache[alias] = info
        return info

    def get_available_dates(self, alias):
        """返回 pred.pkl 中所有可用日期（排序后）。"""
        df = self._load_predictions(alias)
        dates = df.index.get_level_values(0).unique().sort_values()
        return [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates]

    def get_predictions(self, alias, date=None, top_k=30):
        """返回指定日期的 top-K 预测排名。"""
        df = self._load_predictions(alias)
        names = self.data.get_names() if self.data is not None else {}

        # 获取可用日期
        all_dates = df.index.get_level_values(0).unique().sort_values()
        requested_target = pd.Timestamp(date) if date else None

        if requested_target is not None:
            target_date = requested_target
            if target_date not in all_dates:
                # 找最近的可用日期
                fallback_dates = all_dates[all_dates <= target_date]
                if len(fallback_dates) == 0:
                    raise ValueError(f"No prediction data on or before {date}")
                target_date = fallback_dates[-1]
        else:
            target_date = all_dates[-1]

        day_data = df.loc[target_date].sort_values("score", ascending=False)
        top_k = min(top_k, len(day_data))
        metadata = self._prediction_metadata(day_data.iloc[0], target_date)

        stocks = []
        for rank, (instrument, row) in enumerate(day_data.head(top_k).iterrows(), 1):
            instrument = str(instrument).upper()
            code6 = instrument[2:] if instrument.startswith(("SH", "SZ")) else instrument
            stocks.append({
                "rank": rank,
                "instrument": instrument,
                "name": names.get(code6) or names.get(instrument) or instrument,
                "score": round(float(row["score"]), 4),
            })

        all_scores = day_data["score"]
        response = {
            "date": target_date.strftime("%Y-%m-%d"),
            "topK": top_k,
            "stocks": stocks,
            "totalStocks": len(day_data),
            "scoreStats": {
                "mean": round(float(all_scores.mean()), 4),
                "std": round(float(all_scores.std()), 4),
                "min": round(float(all_scores.min()), 4),
                "max": round(float(all_scores.max()), 4),
            },
            **metadata,
        }
        if requested_target is not None and target_date != requested_target:
            response["resolvedFromDate"] = target_date.strftime("%Y-%m-%d")
        return response

    def get_stock_prediction(self, alias, instrument):
        """返回某只股票的 score 时序。"""
        df = self._load_predictions(alias)
        try:
            stock_df = df.xs(instrument, level=1).sort_index()
        except KeyError:
            raise ValueError(f"Instrument '{instrument}' not found in predictions for '{alias}'")

        series = [
            {"date": idx.strftime("%Y-%m-%d"), "score": round(float(row["score"]), 4)}
            for idx, row in stock_df.iterrows()
        ]
        return {"instrument": instrument, "series": series}

    def get_backtest_report(self, alias):
        """返回回测报告关键指标（从 MLflow metrics 读取）。"""
        info = self.get_model_info(alias)
        return {
            "metrics": info["metrics"],
            "params": info["params"],
            "dateRange": info["dateRange"],
        }

    def _predict_or_fallback(self, model, dataset, feature_date, requested_date, alias):
        """尝试在 feature_date 上预测，若结果为空则回退到最近可用交易日的预测。

        Returns:
            Tuple of (pd.DataFrame with column 'score', pd.Timestamp actual_feature_date).
        """
        for attempt in range(2):
            segment = slice(feature_date, feature_date)
            logger.info(
                "Running live prediction for '%s': requested=%s, feature=%s (attempt=%d)",
                alias,
                requested_date.strftime("%Y-%m-%d"),
                feature_date.strftime("%Y-%m-%d"),
                attempt,
            )

            pred = model.predict(dataset, segment=segment)

            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")

            if not pred.empty:
                pred_dates = pred.index.get_level_values(0)
                logger.info(
                    "Predict for '%s' returned %d rows, dates %s ~ %s, target=%s",
                    alias, len(pred), pred_dates.min(), pred_dates.max(), feature_date,
                )
                if feature_date in pred_dates:
                    return pred.loc[feature_date], feature_date
                    return pred.loc[feature_date], feature_date

                available_dates = pred_dates.unique().sort_values()
                logger.warning(
                    "Prediction for '%s': feature_date %s not in results. "
                    "Available dates: %s. Attempting fallback.",
                    alias,
                    feature_date,
                    ", ".join(d.strftime("%Y-%m-%d") for d in available_dates[-5:]),
                )
                # 结果中有其他日期，尝试用当前 segment 能取到的最后一个日期
                if len(available_dates) > 0:
                    fallback_date = available_dates[-1]
                    if fallback_date < feature_date:
                        logger.info(
                            "Fallback for '%s': using %s instead of %s",
                            alias, fallback_date, feature_date,
                        )
                        return pred.loc[fallback_date], fallback_date

            # pred is empty — try previous trading day
            if attempt == 0:
                status = self._get_live_data_status()
                trading_dates = status.get("tradingDates", [])
                feature_str = feature_date.strftime("%Y-%m-%d")
                idx = bisect.bisect_left(trading_dates, feature_str)
                previous_date = trading_dates[idx - 1] if idx > 0 else None
                if previous_date is not None:
                    logger.warning(
                        "Empty prediction for '%s' on %s, falling back to %s",
                        alias, feature_str, previous_date,
                    )
                    feature_date = pd.Timestamp(previous_date)
                    # 需要让 dataset handler 重建以覆盖回退日期
                    self._ensure_handler_data(dataset, target_date=feature_date)
                    continue

            raise ValueError(
                f"No prediction generated for feature date {feature_date.strftime('%Y-%m-%d')}. "
                "The date may be outside the dataset's available data range or is not a trading day."
            )

    def run_prediction(self, alias, date, top_k=30):
        """实时运行模型预测：加载模型 + 数据集，对目标日期调用 predict。

        Args:
            alias: 模型别名
            date: 目标交易日 YYYY-MM-DD
            top_k: 返回排名前 K 的股票

        Returns:
            dict 包含 stocks 排名、scoreStats 等
        """
        requested = pd.Timestamp(date)
        resolved = self._resolve_live_prediction_dates(requested)
        feature_date = resolved["featureDate"]
        requested_date = resolved["requestedDate"]

        with self._lock:
            cached_ds = self._dataset_cache.get(alias)
        if cached_ds is not None:
            handler = getattr(cached_ds, "handler", None)
            if handler is None or not hasattr(handler, "_infer") or handler._infer is None:
                with self._lock:
                    self._dataset_cache.pop(alias, None)
                logger.info("Cleared dataset cache for '%s': cached infer data missing", alias)
            else:
                max_date = handler._infer.index.get_level_values(0).max()
                if feature_date > max_date:
                    with self._lock:
                        self._dataset_cache.pop(alias, None)
                    logger.info(
                        "Cleared dataset cache for '%s': feature date %s > max_date %s",
                        alias,
                        feature_date,
                        max_date,
                    )

        model = self._load_model(alias)
        dataset = self._load_dataset(alias, target_date=feature_date)

        pred, actual_feature_date = self._predict_or_fallback(
            model, dataset, feature_date, requested_date, alias,
        )

        day_data = pred.sort_values("score", ascending=False)
        self._validate_live_prediction_universe(alias, day_data, actual_feature_date)
        self._persist_live_predictions(alias, day_data, requested_date, actual_feature_date, resolved)
        top_k = min(top_k, len(day_data))

        stocks = []
        names = self.data.get_names() if self.data is not None else {}
        for rank, (instrument, row) in enumerate(day_data.head(top_k).iterrows(), 1):
            instrument = str(instrument).upper()
            code6 = instrument[2:] if instrument.startswith(("SH", "SZ")) else instrument
            stocks.append({
                "rank": rank,
                "instrument": instrument,
                "name": names.get(code6) or names.get(instrument) or instrument,
                "score": round(float(row["score"]), 4),
            })

        all_scores = day_data["score"]
        feature_date_str = actual_feature_date.strftime("%Y-%m-%d")
        requested_date_str = requested_date.strftime("%Y-%m-%d")

        date_mapped = resolved["dateMapped"] or actual_feature_date != feature_date
        if actual_feature_date != feature_date:
            response_message = (
                f"目标交易日 {requested_date_str} 的特征数据尚未就绪，"
                f"已使用最近可用数据日 {actual_feature_date.strftime('%Y-%m-%d')} 生成选股结果。"
            )
        else:
            response_message = resolved["message"]

        return {
            "date": requested_date_str,
            "requestedDate": requested_date_str,
            "featureDate": feature_date_str,
            "dateMapped": date_mapped,
            "message": response_message,
            "timingMode": resolved.get("timingMode"),
            "topK": top_k,
            "stocks": stocks,
            "totalStocks": len(day_data),
            "scoreStats": {
                "mean": round(float(all_scores.mean()), 4),
                "std": round(float(all_scores.std()), 4),
                "min": round(float(all_scores.min()), 4),
                "max": round(float(all_scores.max()), 4),
            },
            "source": "live",
            "diagnostics": {
                "latestCalendarDate": resolved["latestCalendarDate"],
                "effectiveLastDate": resolved["effectiveLastDate"],
                "marketEffectiveLastDate": resolved["marketEffectiveLastDate"],
                "syncing": resolved.get("syncing", False),
            },
        }


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _fmt_time(ts_ms):
    if not ts_ms:
        return ""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def _pick(src_dict, keys):
    """从 src_dict 中选取指定 key，支持重命名。结果 dict 使用新 key。"""
    result = {}
    for src_key, dst_key in keys:
        if src_key in src_dict:
            val = src_dict[src_key]
            # 尝试转为数字
            try:
                val = float(val)
                if val == int(val) and "." not in src_dict[src_key]:
                    val = int(val)
            except (ValueError, TypeError):
                pass
            result[dst_key] = val
    return result
