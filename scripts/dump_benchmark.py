"""Download SH000300 (CSI 300) benchmark data and dump to Qlib binary format."""
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

QLIB_DIR = Path.home() / ".quant_master" / "quant_master_data" / "tdx_cn_data"
FEATURES_DIR = QLIB_DIR / "features"
CALENDARS_DIR = QLIB_DIR / "calendars"
INSTRUMENTS_DIR = QLIB_DIR / "instruments"

INDEX_BENCH_URL = (
    "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid=1.{index_code}"
    "&fields1=f1,f2,f3,f4,f5"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
    "&klt=101&fqt=0&beg={begin}&end={end}"
)

FIELDS = ["open", "close", "high", "low", "volume", "amount", "change"]


def download_index(code="000300", begin="20080101", end="20260530"):
    url = INDEX_BENCH_URL.format(index_code=code, begin=begin, end=end)
    logger.info(f"Downloading {code} data from eastmoney...")
    resp = requests.get(url, timeout=30)
    data = resp.json()["data"]["klines"]
    rows = [x.split(",") for x in data]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount", "change"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume", "amount", "change"]:
        df[c] = df[c].astype(float)
    df = df.set_index("date").sort_index()
    logger.info(f"Downloaded {len(df)} rows: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def dump_to_qlib(df, instrument="sh000300"):
    feat_dir = FEATURES_DIR / instrument
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load calendar
    cal_file = CALENDARS_DIR / "day.txt"
    cal = pd.read_csv(cal_file, header=None, parse_dates=[0])
    calendar_list = [pd.Timestamp(d) for d in cal[0].tolist()]

    # Merge data with calendar (fill NaN for missing days)
    cal_df = pd.DataFrame({"date": calendar_list})
    cal_df = cal_df[(cal_df["date"] >= df.index.min()) & (cal_df["date"] <= df.index.max())]
    cal_df.set_index("date", inplace=True)
    merged = df.reindex(cal_df.index)

    # Find start index in calendar
    start_index = calendar_list.index(merged.index.min())
    logger.info(f"Start index: {start_index}, data spans {len(merged)} calendar days")

    for field in FIELDS:
        if field not in merged.columns:
            logger.warning(f"Field {field} not in data, skipping")
            continue

        # Qlib binary format: [start_index, val1, val2, ...]
        values = merged[field].values.astype(np.float32)
        out = np.hstack([np.float32(start_index), values]).astype("<f")

        out_path = feat_dir / f"{field}.day.bin"
        out.tofile(str(out_path))
        logger.info(f"Wrote {out_path} ({len(values)} values from index {start_index})")

    # Update instruments/all.txt
    inst_file = INSTRUMENTS_DIR / "all.txt"
    lines = inst_file.read_text(encoding="utf-8").strip().split("\n")
    existing = {l.strip().split("\t")[0] for l in lines if l.strip()}

    if instrument not in existing:
        start_date = merged.index[0].strftime("%Y-%m-%d")
        end_date = merged.index[-1].strftime("%Y-%m-%d")
        lines.append(f"{instrument}\t{start_date}\t{end_date}")
        inst_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Added {instrument} to all.txt")

    logger.info(f"Done! SH000300 benchmark data dumped to {feat_dir}")


if __name__ == "__main__":
    df = download_index()
    dump_to_qlib(df)
