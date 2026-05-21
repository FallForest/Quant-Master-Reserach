# Get future trading days

> `D.calendar(future=True)` will be used

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# parse instruments, using in quant_master/instruments.
python future_trading_date_collector.py --quant_master_dir ~/.quant_master/quant_master_data/cn_data --freq day
```

## Parameters

- quant_master_dir: quant_master data directory
- freq: value from [`day`, `1min`], default `day`



