# 常用接口维护文档

本文档维护项目里最常用、最容易被重复调用的接口，重点覆盖：

- CN 日线数据同步
- 个人动态股票池更新
- QuantMaster 数据层初始化与读取

统一约定：

- CN 日线数据目录固定使用 `C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data`
- 所有接口默认围绕该目录工作
- 个人动态股票池不是 CSI1000 / CSI300 这种外部成分股池，而是按本地历史数据每日筛选得到

## 1. 常用接口一览

| 接口 | 入口 | 用途 | 备注 |
| --- | --- | --- | --- |
| 数据下载 | `python scripts/get_data.py quant_master_data` | 首次下载或补齐 CN/US 数据 | CN 日线是训练和回测的基础 |
| 日线同步 | `python scripts/data_collector/tdx/collector.py update_data_to_bin` | TDX CN 日线同步、归一化、落 bin | 推荐的日常同步入口 |
| 个人股票池更新 | `python scripts/update_personal_dynamic_pool.py` | 按日期更新个人动态股票池 | 支持单日和区间更新 |
| Python 初始化 | `quant_master.init(...)` | 初始化数据 provider | 训练、特征计算前必须调用 |
| 特征读取 | `quant_master.data.D.features(...)` | 读取特征和筛选候选股票 | 股票池更新内部也依赖它 |

## 2. CN 日线同步接口

### 命令行

```powershell
python scripts/data_collector/tdx/collector.py update_data_to_bin `
  --quant_master_data_1d_dir C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--quant_master_data_1d_dir` | CN 日线统一目录 |
| `--end_date` | 结束日期，开区间 |
| `--delay` | 下载间隔 |
| `--allow_unadjusted` | 是否允许原始价格实验模式 |
| `--adjustment_mode` | `strict` / `raw` / `xdxr` / `adjclose` |
| `--update_personal_pool` | 同步完成后是否顺带更新个人动态股票池 |
| `--personal_pool_date` | 指定单日更新 |
| `--personal_pool_start_date` | 指定区间起始日 |
| `--personal_pool_end_date` | 指定区间结束日 |
| `--skip_names` | 跳过名称文件刷新，适合只同步行情 |
| `--skip_index` | 跳过指数成分股下载/刷新（Yahoo）；TDX 中仅记录为不适用 |
| `--skip_pool` | 跳过个人动态股票池刷新 |

每次成功完成行情落盘和校验后，入口会在数据目录写入原子更新状态文件
`update_manifest.json`，记录下载、标准化、落盘、校验以及名称、指数、股票池阶段，便于调度器重试失败的派生任务。

### 示例

```powershell
python scripts/data_collector/tdx/collector.py update_data_to_bin `
  --quant_master_data_1d_dir C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data `
  --update_personal_pool `
  --personal_pool_date 2026-07-22
```

## 3. 个人动态股票池接口

### CLI

```powershell
python scripts/update_personal_dynamic_pool.py `
  --provider-uri C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data `
  --date 2026-07-22
```

可调参数包括 `--min-price`、`--max-price`、`--min-history-days`、`--min-average-amount`、
`--activity-lookback-days`、`--min-average-abs-return`、`--boards` 以及股票池数量上下限。

### Python

```python
from quant_master.data.personal_dynamic_pool import update_personal_dynamic_pool

result = update_personal_dynamic_pool(
    provider_uri=r"C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data",
    update_date="2026-07-22",
)
```

### 默认筛选规则

| 规则 | 默认值 |
| --- | --- |
| 板块 | 主板 + 创业板 + 科创板 |
| 排除 | 北交所 |
| 价格区间 | `2 <= close <= 40` |
| 历史长度 | `>= 120` 个交易日 |
| 流动性 | 20 日均成交额 `>= 5000 万` |
| 活跃度 | 前 20 日平均绝对日收益 `>= 0.8%` |

筛选在收盘后执行：当日收盘价、成交量和成交额可用，滚动历史条件只使用前一交易日及更早数据。
输出文件是 `instruments/personal_dynamic_pool.txt`，因此训练 handler 可以直接写
`instruments: personal_dynamic_pool`。建议先回填训练区间，再在每天数据同步后更新最新交易日：

```powershell
# 首次回填（区间端点包含在内）
python scripts/update_personal_dynamic_pool.py `
  --provider-uri C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data `
  --start-date 2018-01-01 --end-date 2026-08-07

# 日常增量更新：TDX 同步日线，再重算最新交易日
python scripts/data_collector/tdx/collector.py update_data_to_bin `
  --quant_master_data_1d_dir C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data `
  --allow_unadjusted --update_personal_pool
```

只同步行情时可跳过派生任务：

```powershell
python scripts/data_collector/tdx/collector.py update_data_to_bin `
  --quant_master_data_1d_dir C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data `
  --allow_unadjusted --skip_names --skip_pool
```

`--allow_unadjusted` 只适用于当前 TDX 原始价格数据；如果接入带可靠复权因子的源数据，应使用
`adjustment_mode=xdxr` 或 `adjclose`，并移除该参数。股票池是训练候选集，实盘下单仍需复核 ST、停牌、涨跌停、权限和 100 股一手的可买性。

## 4. Python 常用入口

### 初始化数据层

```python
import quant_master

quant_master.init(
    provider_uri=r"C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data",
    region="cn",
)
```

### 读取特征

```python
from quant_master.data import D

df = D.features(D.instruments("all"), ["$close", "$volume", "$amount"], start_time="2026-07-01", end_time="2026-07-22")
```

## 5. 维护约定

当你新增、删除或改动以下内容时，请同步更新这份文档：

- CLI 参数名
- 默认值
- 数据目录
- 个人股票池筛选规则
- 输出文件名

建议把变化分成两类记录：

- 破坏性变化：参数语义、目录结构、文件格式变化
- 非破坏性变化：新增可选参数、增加日志、增加辅助脚本
