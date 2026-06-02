
- [Download QuantMaster Data](#Download-QuantMaster-Data)
  - [Download CN Data](#Download-CN-Data)
  - [Download US Data](#Download-US-Data)
  - [Download CN Simple Data](#Download-CN-Simple-Data)
  - [Help](#Help)
- [Using in QuantMaster](#Using-in-QuantMaster)
  - [US data](#US-data)
  - [CN data](#CN-data)


## Download QuantMaster Data


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

### Download CN Simple Data

```bash
python get_data.py quant_master_data --name quant_master_data_simple --target_dir ~/.quant_master/quant_master_data/tdx_cn_data --region cn
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

## Use Crowd Sourced Data
The is also a [crowd sourced version of quant_master data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/quant_master_bin.tar.gz
tar -zxvf quant_master_bin.tar.gz -C ~/.quant_master/quant_master_data/tdx_cn_data --strip-components=2
```
