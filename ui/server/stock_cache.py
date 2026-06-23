"""股票列表预计算缓存：避免每次浏览都遍历全量 bin 文件。

将每只股票的最新价、涨跌幅、成交量预计算后写入 JSON，
/browser/stocks 改读该缓存，配合内存 TTL 减少重复 I/O。
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_log = logging.getLogger(__name__)

# 并行构建缓存的线程数，匹配 sync.py 的 NUM_WORKERS
_NUM_WORKERS = min(16, (os.cpu_count() or 1) * 2)

# 内存缓存（进程级）
_cache: dict = {"stocks": None, "ts": 0.0}
CACHE_TTL = 900  # 秒(15分钟)，大幅降低重复文件 I/O，数据同步完成后会自动重建缓存


def _cache_dir():
    from .config import LIVE_DATA_DIR
    return LIVE_DATA_DIR / "stock_summary"


def _cache_file():
    return _cache_dir() / "latest.json"


def _build_stock_item(sym: str, start: str, end: str, names: dict, data_dir_str: str) -> dict:
    """处理单只股票的行情摘要（供 ThreadPoolExecutor 调用）。"""
    from .helpers import _last_non_nan, _prev_non_nan_nonzero
    from .datadir import create_data_dir

    code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
    name = names.get(code6, "")
    item: dict = {"symbol": sym, "name": name, "startDate": start, "endDate": end}

    # 每个 worker 持有独立的 DataDir 实例，避免锁竞争
    data = create_data_dir(data_dir_str)

    # 最新收盘价 & 涨跌
    _, close_vals = data.read_field(sym, "close", "day")
    if close_vals is not None and len(close_vals) > 0:
        raw_c = _last_non_nan(close_vals)
        if raw_c is not None:
            c = round(raw_c, 2)
            item["close"] = c
            prev = _prev_non_nan_nonzero(close_vals, len(close_vals) - 1)
            if prev is not None:
                item["change"] = round(c - prev, 2)
                item["changePct"] = round((c - prev) / prev * 100, 2)

    # 成交量
    _, vol_vals = data.read_field(sym, "volume", "day")
    if vol_vals is not None and len(vol_vals) > 0:
        raw_v = _last_non_nan(vol_vals)
        if raw_v is not None:
            item["volume"] = int(raw_v)

    return item


def build_stock_summary(data) -> list[dict] | None:
    """并行遍历全量股票，预计算最新行情摘要，写入 JSON 文件。

    使用 ThreadPoolExecutor 多线程读取 .bin 文件，大幅缩短构建时间。
    返回值供调用方（启动时的 warmup）同时刷新内存缓存。
    """
    instruments = data.get_instruments()
    if not instruments:
        _log.warning("build_stock_summary: no instruments found")
        return None
    names = data.get_names()

    # 提取 data_dir 字符串传递给 worker，避免跨线程共享 DataDir 对象
    data_dir_str = data.data_dir

    result: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as pool:
        futs = {}
        for sym, start, end in instruments:
            fut = pool.submit(_build_stock_item, sym, start, end, names, data_dir_str)
            futs[fut] = sym

        for fut in as_completed(futs):
            try:
                result.append(fut.result())
            except Exception:
                sym = futs[fut]
                _log.warning("Failed to build stock summary for %s", sym, exc_info=True)

    elapsed = time.time() - t0
    _log.info("Stock summary built %d stocks in %.1fs (%d workers)", len(result), elapsed, _NUM_WORKERS)

    # 落地到 JSON 文件
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "latest.json"
    payload = {"stocks": result, "count": len(result), "builtAt": time.time()}
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 刷新内存缓存
    _cache["stocks"] = result
    _cache["ts"] = time.time()

    _log.info("Stock summary cache file written: %s", cache_file)
    return result


def load_stock_summary() -> list[dict] | None:
    """返回缓存的股票摘要列表。

    优先级：内存 → 文件。文件不存在或无效时返回 None，
    调用方 (/browser/stocks) 走降级路径逐个读 bin。
    """
    now = time.time()

    # 1) 内存命中且未过期
    if _cache["stocks"] is not None and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["stocks"]

    # 2) 从文件加载
    cache_file = _cache_file()
    if cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            stocks = raw.get("stocks", [])
            _cache["stocks"] = stocks
            _cache["ts"] = now
            return stocks
        except Exception:
            _log.warning("Stock summary cache file damaged, re-building...", exc_info=True)
            cache_file.unlink(missing_ok=True)

    return None


def invalidate_cache() -> None:
    """使内存缓存失效，下次请求重新读取文件。"""
    _cache["ts"] = 0.0
