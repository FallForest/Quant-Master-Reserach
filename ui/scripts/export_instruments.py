"""从 quant_master 数据导出股票列表为 JSON（含最新价格）

优先从 instruments/names.txt 读取名称（由 update_data_to_bin 自动生成），
若不存在则名称留空。
价格从 features/<symbol>/close.day.bin 读取（最后一个值为最新收盘价）。
"""
import json, sys, os
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.quant_master/quant_master_data/cn_data")
OUT = os.path.join(os.path.dirname(__file__), "..", "public", "instruments.json")

# 读取名称映射 (code -> name)
names_path = os.path.join(DATA_DIR, "instruments", "names.txt")
code_name = {}
if os.path.exists(names_path):
    with open(names_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                code_name[parts[0]] = parts[1]
    print(f"Loaded {len(code_name)} stock names from {names_path}")
else:
    print(f"Warning: {names_path} not found, names will be empty. Run update_data_to_bin first.")

# 读 instruments
inst_file = os.path.join(DATA_DIR, "instruments", "all.txt")
if not os.path.exists(inst_file):
    print(f"ERROR: {inst_file} not found", file=sys.stderr)
    sys.exit(1)


def read_last_bin(symbol, field):
    """读取 bin 文件最后一个值（最新数据）。"""
    d = os.path.join(DATA_DIR, "features", symbol.lower())
    for suffix in ["day.bin", "1min.bin"]:
        fpath = os.path.join(d, f"{field}.{suffix}")
        if os.path.exists(fpath):
            try:
                data = np.fromfile(fpath, dtype="<f4")
                # 第一个值是日历索引，后面是实际数据
                if len(data) > 1:
                    val = float(data[-1])
                    if not np.isnan(val):
                        return val
            except Exception:
                pass
    return None


stocks = []
matched = 0
with open(inst_file) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        sym, start, end = parts[0], parts[1], parts[2]
        # 提取6位数字代码: SZ000001 -> 000001, SH600519 -> 600519
        code6 = sym[2:] if len(sym) == 8 and sym[:2] in ("SZ", "SH", "BJ") else sym
        name = code_name.get(code6, "")
        if name:
            matched += 1

        # 读取最新价格
        close = read_last_bin(sym, "close")
        prev_close = None
        change = None
        change_pct = None
        if close is not None:
            # 读倒数第二个 close 来算涨跌
            d = os.path.join(DATA_DIR, "features", sym.lower())
            for suffix in ["day.bin", "1min.bin"]:
                fpath = os.path.join(d, f"close.{suffix}")
                if os.path.exists(fpath):
                    try:
                        data = np.fromfile(fpath, dtype="<f4")
                        if len(data) > 2:
                            prev = float(data[-2])
                            if not np.isnan(prev) and prev != 0:
                                prev_close = prev
                                change = round(close - prev, 2)
                                change_pct = round((close - prev) / prev * 100, 2)
                    except Exception:
                        pass
                    break

        volume = read_last_bin(sym, "volume")

        stock = {
            "symbol": sym,
            "name": name,
            "startDate": start,
            "endDate": end,
        }
        if close is not None:
            stock["close"] = round(close, 2)
        if change is not None:
            stock["change"] = change
            stock["changePct"] = str(change_pct)
        if volume is not None:
            stock["volume"] = int(volume)

        stocks.append(stock)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(stocks, f, ensure_ascii=False)

print(f"Exported {len(stocks)} stocks ({matched} with names) -> {OUT}")
