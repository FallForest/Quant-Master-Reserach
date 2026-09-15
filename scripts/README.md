
- [Download QuantMaster Data](#Download-QuantMaster-Data)
  - [Download CN Data](#Download-CN-Data)
  - [Download US Data](#Download-US-Data)
  - [Help](#Help)
- [Using in QuantMaster](#Using-in-QuantMaster)
  - [US data](#US-data)
  - [CN data](#CN-data)


## Download QuantMaster Data

常用接口总览见 [docs/common_interfaces.md](../docs/common_interfaces.md).


### Download CN Data

```bash
# daily data
python get_data.py quant_master_data --target_dir ~/.quant_master/quant_master_data/tdx_cn_data --region cn

# 1min  data (Optional for running non-high-frequency strategies)
python get_data.py quant_master_data --target_dir ~/.quant_master/quant_master_data/cn_data_1min --region cn --interval 1min
```

The current unified CN daily data directory is `~/.quant_master/quant_master_data/tdx_cn_data`
(Windows runtime path: `C:\Users\15728\.quant_master\quant_master_data\tdx_cn_data`).

### Download US Data


```bash
python get_data.py quant_master_data --target_dir ~/.quant_master/quant_master_data/us_data --region us
```

### Help

```bash
python get_data.py quant_master_data --help
```

## Using in QuantMaster
> For more information: https://quant_master.readthedocs.io/en/latest/start/initialization.html


### US data

> Need to download data first: [Download US Data](#Download-US-Data)

```python
import quant_master
from quant_master.config import REG_US
provider_uri = "~/.quant_master/quant_master_data/us_data"  # target_dir
quant_master.init(provider_uri=provider_uri, region=REG_US)
```

### CN data

> Need to download data first: [Download CN Data](#Download-CN-Data)

```python
import quant_master
from quant_master.constant import REG_CN

provider_uri = "~/.quant_master/quant_master_data/tdx_cn_data"  # current unified CN target_dir
quant_master.init(provider_uri=provider_uri, region=REG_CN)
```

### Personal dynamic stock pool

The daily point-in-time pool is generated from the unified CN daily dataset and can be used by any
handler through `instruments: personal_dynamic_pool`. The default pool requires a CNY 2–40 price,
120 days of history, 20-day average amount of at least CNY 50 million, and a 20-day average absolute
daily return of at least 0.8%. See [docs/common_interfaces.md](../docs/common_interfaces.md)
for the filter contract and the update commands. The short version is:

```bash
python scripts/update_personal_dynamic_pool.py \
  --provider-uri ~/.quant_master/quant_master_data/tdx_cn_data \
  --start-date 2018-01-01 --end-date 2026-08-07
```

## Use Crowd Sourced Data
The is also a [crowd sourced version of quant_master data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/quant_master_bin.tar.gz
tar -zxvf quant_master_bin.tar.gz -C ~/.quant_master/quant_master_data/tdx_cn_data --strip-components=2
```
