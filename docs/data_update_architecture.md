# 数据更新接口与解耦建议

项目的数据更新可以按“源数据、标准化、QuantMaster 落盘、派生资产、训练读取”五层理解。统一 CN 日线目录是：
`C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data`。

## 接口清单

| 层 | 入口 | 输入 | 输出/副作用 | 适用场景 |
| --- | --- | --- | --- | --- |
| 首次数据包 | `python scripts/get_data.py quant_master_data` | 数据集名、地区、频率、目标目录 | 下载并解压 `calendars/`、`instruments/`、`features/` | 初始化环境、恢复基线 |
| TDX 下载 | `python scripts/data_collector/tdx/collector.py download_data` | 日期、频率、股票范围 | `scripts/data_collector/tdx/source_*/*.csv` | 只抓原始行情，调试或分阶段运行 |
| TDX 标准化 | `... tdx/collector.py normalize_data` | source CSV、调整模式 | `normalize_*/*.csv` | 调整因子已在 source 中准备好时分阶段运行 |
| TDX 增量同步 | `... tdx/collector.py update_data_to_bin` | 统一目录、日期、下载范围 | 更新二进制、日历、`instruments/all.txt`、名称和指数成分 | CN 日常行情同步 |
| Yahoo 增量同步 | `... yahoo/collector.py update_data_to_bin` | 统一目录、截止日期 | 同上 | Yahoo 作为替代日线源；默认不适合作为个人池源 |
| 二进制全量落盘 | `python scripts/dump_bin.py dump_all` | 标准化 CSV | 新建 QuantMaster 二进制目录 | 新数据集或重建 |
| 二进制增量落盘 | `python scripts/dump_bin.py dump_update` | 标准化 CSV、已有目录 | 追加 features、更新日历和 instruments | 分阶段同步 |
| 指数成分 | `python scripts/data_collector/{cn_index,us_index}/collector.py` | 指数名、目标目录 | `instruments/csi*.txt` 等 PIT 区间文件 | CSI/海外指数市场 |
| 个人动态池 | `python scripts/update_personal_dynamic_pool.py` | 统一目录、单日或日期区间 | `instruments/personal_dynamic_pool.txt` 和 metadata | 训练候选集 |
| 池联动 | `--update_personal_pool` | 日线同步成功后的统一目录 | 重算最新池或指定区间 | 日常自动刷新 |
| 健康检查 | `python scripts/check_data_health.py` | 二进制目录或 CSV | 缺失、突变、复权和 universe 报告 | 更新后验收 |
| 训练读取 | `D.instruments(...)`、`D.features(...)` | market、表达式、日期 | DataFrame 或按日期有效的 instruments | 训练、推理、回测 |

## 推荐流程

首次安装先用 `get_data.py` 准备基线；CN 个人池场景日常执行 TDX 的 `update_data_to_bin`，成功校验后再更新个人动态池；训练通过 `instruments: personal_dynamic_pool` 读取 PIT 区间。Yahoo 可作为替代行情源，但必须先在标准化数据中补齐可靠的 `amount` 字段，才能启用个人池联动。指数成分更新和健康检查属于独立派生任务，可按需执行。

```text
source API → source CSV → normalize → dump/verify → names/index files
                                             └────→ personal_dynamic_pool
```

个人池只依赖“日线二进制已经成功落盘且校验通过”，不会参与下载、标准化或二进制写入。TDX/Yahoo 两个 collector 通过 `scripts/data_collector/pool_integration.py` 调用同一个联动边界；如果源数据没有 `amount`，池更新应被视为不满足前置条件。

## 当前耦合与建议

当前最高耦合点是两个 collector 的长函数：下载、清理临时文件、标准化、落盘、校验、名称、指数和股票池都在一个命令中完成。已拆出的低风险边界是个人池联动；原有命令和参数保持兼容。

### UI 同步链路

UI 使用 `ui/server/sync.py` 的专用增量同步实现，不直接调用命令行 collector。它已经在 HTTP 层与行情任务解耦：`/api/pipeline/trigger` 只启动后台任务，页面通过 `/api/pipeline/status` 轮询；行情抓取、二进制写入和校验也有独立进度阶段。

行情校验完成后，股票摘要 JSON 缓存属于独立派生任务，后台单独重建并使用临时文件原子替换。此时 `syncing` 会结束，状态中的 `cacheRefresh.running` 继续反映缓存重建，浏览器可继续使用上一份完整缓存。TDX 抓取并发数可通过 `QUANT_MASTER_SYNC_WORKERS` 配置，默认 12，范围限制为 1 到 32。

因此 UI 当前是“行情任务与缓存任务解耦”，但仍然是“每只股票一次历史 K 线请求”的同步策略；5208 只股票时，网络请求数量仍是主要耗时来源。后续若要继续缩短抓取时间，应引入批量行情/历史数据源或按数据源能力拆出专门的批量下载器，而不是继续把线程数无限增大。

后续建议按优先级处理：

1. **已实现：** `update_data_to_bin` 支持 `--skip_names`、`--skip_index`、`--skip_pool`，避免只更新行情时触发不必要的网络请求；Yahoo 指数下载也受 `skip_index` 控制。
2. **已实现：** 将下载、标准化、落盘、校验及派生阶段写入数据目录的 `update_manifest.json`，采用临时文件替换，调度器可据此重试失败的派生任务。
3. **中优先级：** 把 TDX/Yahoo 的公共编排抽到 `BaseRun.update_pipeline()`，源适配器只实现 fetch/normalize；保留现有 `update_data_to_bin` 作为兼容外壳。
4. **中优先级：** 统一所有命令的日期语义（开始日期包含、结束日期开区间），并统一 `day` 与 `1d` 的频率映射。
5. **低优先级：** 将指数成分、名称同步、健康检查做成独立可重试任务；它们失败时不应回滚已验证的行情二进制。

## 性能建议

- 增量池只更新最新交易日；历史回填才使用日期区间。
- 池计算已按日期分块并关闭磁盘 dataset cache，避免把大横截面缓存成过期结果。
- 下载、标准化和落盘目前分别有并发控制；不要盲目提高 worker 数，TDX/Yahoo 限流和磁盘写入通常才是瓶颈。
- 每次落盘后清理 instrument memory cache；训练进程长驻时应显式重新初始化或清缓存。
