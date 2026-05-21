# Collect Crypto Data

> *Please pay **ATTENTION** that the data is collected from [Coingecko](https://www.coingecko.com/en/api) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://quant_master.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-quant_master-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Usage of the dataset
> *Crypto dataset only support Data retrieval function but not support backtest function due to the lack of OHLC data.*

## Collector Data


### Crypto Data

#### 1d from Coingecko

```bash

# download from https://api.coingecko.com/api/v3/
python collector.py download_data --source_dir ~/.quant_master/crypto_data/source/1d --start 2015-01-01 --end 2021-11-30 --delay 1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.quant_master/crypto_data/source/1d --normalize_dir ~/.quant_master/crypto_data/source/1d_nor --interval 1d --date_field_name date

# dump data
cd quant_master/scripts
python dump_bin.py dump_all --data_path ~/.quant_master/crypto_data/source/1d_nor --quant_master_dir ~/.quant_master/quant_master_data/crypto_data --freq day --date_field_name date --include_fields prices,total_volumes,market_caps

```

### using data

```python
import quant_master
from quant_master.data import D

quant_master.init(provider_uri="~/.quant_master/quant_master_data/crypto_data")
df = D.features(D.instruments(market="all"), ["$prices", "$total_volumes","$market_caps"], freq="day")
```


### Help
```bash
python collector.py collector_data --help
```

## Parameters

- interval: 1d
- delay: 1
