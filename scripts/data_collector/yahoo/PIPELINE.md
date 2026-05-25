# Yahoo Finance 数据更新链路

## 概览

入口: `scripts/data_collector/yahoo/collector.py` → `Run.update_data_to_bin()`

功能: 从 Yahoo Finance 增量下载 A 股日线数据，标准化后写入 quant_master 二进制格式。

## 流程

```
download_data  →  normalize  →  dump_bin  →  verify  →  index_composition
     ↓               ↓             ↓            ↓              ↓
 source/*.csv    normalize/*.csv  features/    integrity     instruments/
                                   calendars/   check         (CSI100/300)
                                   instruments/
```

### 1. 清理残留
- 清空 `source/` 和 `normalize/` 目录
- 防止上次中断留下脏数据

### 2. 初始化数据集
- 如果 `quant_master_data_1d_dir` 不存在 quant_master 数据，自动下载初始数据集
- 跳过条件: `exists_skip=True` 且目录已存在

### 3. 确定日期范围
- 读取 `calendars/day.txt` 最后一行 → `last_cal_date`
- `trading_date = last_cal_date - 1 day` (重叠日期，用于增量衔接)
- `end_date = 默认 last_cal_date` (Yahoo 下载的开区间上界)

### 4. 下载 (Yahoo → source/)
- 并发下载 ~5200 只股票的日线数据
- 每只股票生成 `source/<symbol>.csv`
- 同时下载 CSI300/CSI100/CSI500 指数数据
- 输出列: `date, open, high, low, close, volume, adjclose, symbol`

### 5. 标准化 (source/ → normalize/)
- 对齐交易日历、复权因子计算、收益率计算
- Extend 模式: 与已有二进制数据交叉比对，移除重叠日期
- 输出列: `date, open, high, low, close, volume, factor, change, symbol`

### 6. 写入二进制 (normalize/ → cn_data/)
- 增量追加到 `features/<symbol>/open.day.bin` 等文件
- 更新 `calendars/day.txt` (追加新交易日)
- 清理超过 180 天未交易的 instruments，备份到 `instruments/all.txt.bak.YYYYMMDD`
- 更新 `instruments/all.txt`

### 7. 校验
- 日历最后日期匹配预期
- instruments 无过期条目
- 随机抽样 10 个 feature 目录检查完整性

### 8. 清理中间文件
- 删除 `source/` 和 `normalize/` 下所有 CSV

### 9. 指数成分股
- 更新 CSI100、CSI300 成分股列表到 `instruments/`

## 可配置参数

### Run 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_workers` | int | 8 | 并发线程数。下载阶段实际使用 `min(max(max_workers, 4), 12)` |
| `region` | str | CN | 地区，可选 CN/US/BR |
| `interval` | str | 1d | 频率，仅 1d 支持 update_data_to_bin |

### update_data_to_bin 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `quant_master_data_1d_dir` | str | **必填** | quant_master 二进制数据目录 |
| `end_date` | str | None | 更新截止日期 (开区间)。默认自动取日历最后一天 |
| `delay` | float | 0.1 | 每次请求间隔秒数，防止被 Yahoo 限流 |
| `check_data_length` | int | None | 数据长度阈值，低于此值的股票会重试下载 |
| `exists_skip` | bool | False | 如果数据集已存在，跳过初始下载 |

### CLI 调用

```bash
# 基本用法：增量更新到最新
python scripts/data_collector/yahoo/collector.py update_data_to_bin \
  --quant_master_data_1d_dir ~/.quant_master/quant_master_data/cn_data

# 指定截止日期
python scripts/data_collector/yahoo/collector.py update_data_to_bin \
  --quant_master_data_1d_dir ~/.quant_master/quant_master_data/cn_data \
  --end_date 2026-05-22

# 调整并发和延迟
python scripts/data_collector/yahoo/collector.py update_data_to_bin \
  --quant_master_data_1d_dir ~/.quant_master/quant_master_data/cn_data \
  --max_workers 8 \
  --delay 0.2
```

## 文件结构

```
scripts/data_collector/yahoo/
├── collector.py          # 主程序
├── source/               # 中间产物: Yahoo 原始 CSV (自动清理)
└── normalize/            # 中间产物: 标准化 CSV (自动清理)

<quant_master_data_1d_dir>/
├── calendars/day.txt     # 交易日历
├── features/
│   ├── sh600000/         # 每只股票一个目录
│   │   ├── open.day.bin  # <f4 平坦数组, 首元素为日历索引
│   │   ├── close.day.bin
│   │   ├── high.day.bin
│   │   ├── low.day.bin
│   │   ├── volume.day.bin
│   │   ├── factor.day.bin
│   │   └── change.day.bin
│   └── ...
├── instruments/
│   ├── all.txt           # symbol\tstart_date\tend_date
│   └── all.txt.bak.*     # 被清理 instruments 的备份
└── _latest_map_cache.pkl # _latest_map 缓存 (自动管理)
```

## UI 参数建议

最小表单:
- **数据目录** (必填): 路径选择器，指向 quant_master 数据目录
- **截止日期** (可选): 日期选择器，默认"更新到最新"

高级选项:
- **并发数**: 滑块 1-16，默认 8
- **请求延迟**: 数字输入 0-2s，默认 0.1s
- **数据长度校验**: 数字输入，留空表示不校验
