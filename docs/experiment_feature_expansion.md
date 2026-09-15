# 实验特征扩容参数

本轮扩容按以下优先级实施：分钟线特征、可靠复权/XDXR、行业/指数暴露。目标是增加新的信息来源和可审计的数据契约，而不是继续为 Alpha158 叠加日线窗口。

## 参数面

| 层 | 入口 | 新参数 | 默认行为 |
| --- | --- | --- | --- |
| 分钟线 | `HighFreqGeneralHandler` | `lags`、`feature_families`、`feature_windows` | 保留当前 bar 与前一交易日同刻快照；新增特征族默认关闭 |
| 复权 | `TdxNormalize` / TDX collector CLI | `adjustment_mode` | 日线使用 `strict`，没有上游 `adjclose`/XDXR 时拒绝生成伪复权数据 |
| 暴露 | `TranscendenceAlpha` | `index_exposures`、`industry_exposures`、`exposure_windows` | 暴露默认关闭，保持原工作流列宽不变 |

所有 CN 日线仍使用统一目录：

```text
C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data
```

## 1. 分钟线特征

`HighFreqGeneralHandler` 支持两类正交扩展：

- `lags`：以 bar 数表示的当前/历史快照；
- `feature_families`：`momentum`、`realized_volatility`、`volume_distribution`、`intraday_shape`；
- `feature_windows`：按特征族配置 bar 窗口，不把分钟窗口误写成交易日窗口。

5 分钟线示例：

```yaml
handler:
  class: HighFreqGeneralHandler
  module_path: quant_master.contrib.data.highfreq_handler
  kwargs:
    instruments: csi300
    freq: 5min
    day_length: 48
    columns: [$open, $high, $low, $close, $vwap]
    lags: [0, 6, 12, 48]
    feature_families:
      - momentum
      - realized_volatility
      - volume_distribution
      - intraday_shape
    feature_windows:
      momentum: [3, 6, 12]
      realized_volatility: [6, 12, 24]
      volume_distribution: [6, 12, 48]
      intraday_shape: [6, 12, 24]
```

窗口与 lag 只接受非负或正整数，并拒绝重复项、布尔值和数字字符串。

## 2. 可靠复权/XDXR

`adjustment_mode` 的语义如下：

| 模式 | 行为 | 用途 |
| --- | --- | --- |
| `strict` | 必须有上游提供的 `adjclose`/XDXR 序列，否则失败 | 默认研究与回测 |
| `xdxr` | 显式要求并应用上游 PIT `adjclose`/XDXR | 复权实验 |
| `adjclose` | `xdxr` 的兼容别名 | 已有 adjusted-close 数据源 |
| `raw` | 明确生成 `factor=1` 的原始价格数据 | 仅限隔离诊断，不应与复权实验混比 |

纯 TDX K 线下载不包含可靠 XDXR/`adjclose`。因此 `update_data_to_bin` 不会假装构造复权因子；需要先在 source CSV 中合并 point-in-time 的 `adjclose`/XDXR，再执行：

```powershell
python scripts/data_collector/tdx/collector.py normalize_data `
  --source_dir <PIT_XDXR_SOURCE_DIR> `
  --normalize_dir <NORMALIZED_DIR> `
  --adjustment_mode xdxr
```

复权因子必须有限且大于零，头部缺失、非正 close、非法因子都会失败。价格按 factor 重标，成交量按 factor 反向重标。

## 3. 行业/指数暴露

```yaml
handler:
  class: TranscendenceAlpha
  module_path: quant_master.contrib.data.transcendence_handler
  kwargs:
    instruments: csi300
    benchmark: SH000300
    index_exposures: [SH000300, SH000905]
    industry_exposures:
      bank: SH000951
      technology: SZ399608
    exposure_windows: [20, 60]
```

每个配置项生成对应窗口的超额收益和 beta。`industry_exposures` 表示“个股对行业指数收益的暴露代理”，不是个股的 PIT 行业归属。所有引用指数必须已存在于本地 provider，指数自身的成分与历史口径也必须可审计。

## 建议实验顺序

1. 固定日线基线和模型种子，只打开分钟线轴，先比较 lag 快照，再加入四个分钟特征族。
2. 固定第 1 步胜出配置，使用同一份上游 PIT XDXR 数据比较 `strict/xdxr`；`raw` 只作异常诊断。
3. 固定前两步，依次加入宽基指数暴露、单个行业指数代理、多个行业指数代理。
4. 每次只改变一个轴，记录实际特征列名、provider URI、复权模式、窗口和指数代码，避免把数据版本差异误判为模型增益。

相关可复制片段位于：

- `examples/benchmarks/LightGBM/highfreq_general_handler_lags_example.yaml`
- `examples/benchmarks/Transcendence/configs/feature_exposure_example.yaml`
