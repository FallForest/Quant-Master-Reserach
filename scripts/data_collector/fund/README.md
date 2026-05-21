# Collect Fund Data

> *Please pay **ATTENTION** that the data is collected from [天天基金网](https://fund.eastmoney.com/) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://quant_master.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-quant_master-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data


### CN Data

#### 1d from East Money

```bash

# download from eastmoney.com
python collector.py download_data --source_dir ~/.quant_master/fund_data/source/cn_data --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.quant_master/fund_data/source/cn_data --normalize_dir ~/.quant_master/fund_data/source/cn_1d_nor --region CN --interval 1d --date_field_name FSRQ

# dump data
cd quant_master/scripts
python dump_bin.py dump_all --data_path ~/.quant_master/fund_data/source/cn_1d_nor --quant_master_dir ~/.quant_master/quant_master_data/cn_fund_data --freq day --date_field_name FSRQ --include_fields DWJZ,LJJZ

```

### using data

```python
import quant_master
from quant_master.data import D

quant_master.init(provider_uri="~/.quant_master/quant_master_data/cn_fund_data")
df = D.features(D.instruments(market="all"), ["$DWJZ", "$LJJZ"], freq="day")
```


### Help
```bash
pythono collector.py collector_data --help
```

## Parameters

- interval: 1d
- region: CN

## 免责声明

本项目仅供学习研究使用，不作为任何行为的指导和建议，由此而引发任何争议和纠纷，与本项目无任何关系
