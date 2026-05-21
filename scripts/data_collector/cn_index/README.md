# CSI300/CSI100/CSI500 History Companies Collection

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# parse instruments, using in quant_master/instruments.
python collector.py --index_name CSI300 --quant_master_dir ~/.quant_master/quant_master_data/cn_data --method parse_instruments

# parse new companies
python collector.py --index_name CSI300 --quant_master_dir ~/.quant_master/quant_master_data/cn_data --method save_new_companies

# index_name support: CSI300, CSI100, CSI500
# help
python collector.py --help
```

