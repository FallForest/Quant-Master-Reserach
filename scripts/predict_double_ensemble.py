"""
DoubleEnsemble prediction for May 25 — ALL stocks (5200+).
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import quant_master as qm
from quant_master.contrib.data.handler import Alpha158
from quant_master.data.dataset import DatasetH
from quant_master.contrib.model.double_ensemble import DEnsembleModel


def main():
    provider_uri = str(project_root / ".qmData" / "cn_data")
    qm.init(provider_uri=provider_uri, region="cn")

    # Read all active stocks (end_date = 2026-05-22)
    inst_path = project_root / ".qmData" / "cn_data" / "instruments" / "all.txt"
    stocks = []
    with open(inst_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2] == "2026-05-22":
                stocks.append(parts[0])

    print(f"[INFO] Active stocks: {len(stocks)}")

    target_date = "2026-05-22"  # latest data → scores for May 25 (Mon)
    print(f"[INFO] Target: {target_date} → valid for 2026-05-25 (Monday)")
    print(f"[INFO] Train: 2018-2023, Valid: 2024-2025H1")

    handler = Alpha158(
        start_time="2018-01-01",
        end_time=target_date,
        fit_start_time="2018-01-01",
        fit_end_time="2023-12-31",
        instruments=stocks,
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ["2018-01-01", "2023-12-31"],
            "valid": ["2024-01-01", "2025-06-30"],
            "test": [target_date, target_date],
        },
    )

    model = DEnsembleModel(
        base_model="gbm",
        loss="mse",
        num_models=3,
        enable_sr=True,
        enable_fs=True,
        alpha1=1,
        alpha2=1,
        bins_sr=10,
        bins_fs=5,
        decay=0.5,
        sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
        sub_weights=[1, 1, 1],
        epochs=20,
        colsample_bytree=0.8879,
        learning_rate=0.2,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=20,
        verbosity=-1,
    )

    print("\n[INFO] Training DoubleEnsemble on all stocks...")
    model.fit(dataset)

    print(f"\n[INFO] Predicting for {target_date}...")
    pred = model.predict(dataset, segment="test")

    if isinstance(pred.index, pd.MultiIndex):
        pred_df = pred.reset_index()
        pred_df.columns = ["datetime", "instrument", "score"]
        target_preds = pred_df[pred_df["datetime"] == target_date].copy()
    else:
        target_preds = pd.DataFrame({"instrument": pred.index, "score": pred.values})

    if target_preds.empty:
        print("[WARN] No predictions!")
        return

    target_preds = target_preds.sort_values("score", ascending=False).reset_index(drop=True)
    target_preds["rank"] = range(1, len(target_preds) + 1)

    print(f"\n{'='*70}")
    print(f"  DoubleEnsemble Top 50 — 全市场 — 2026-05-25 (周一)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Stock':<12} {'Score':>10}")
    print(f"{'-'*30}")
    for _, row in target_preds.head(50).iterrows():
        print(f"{row['rank']:<6} {row['instrument']:<12} {row['score']:>10.6f}")

    print(f"\nBottom 10:")
    print(f"{'-'*30}")
    for _, row in target_preds.tail(10).iterrows():
        print(f"{row['rank']:<6} {row['instrument']:<12} {row['score']:>10.6f}")

    # Save
    output_path = project_root / "artifacts" / "double_ensemble_pred_2026-05-25_all.csv"
    output_path.parent.mkdir(exist_ok=True)
    target_preds.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved {len(target_preds)} stocks to {output_path}")

    print(f"\n[STATS] Score distribution:")
    print(f"  Mean={target_preds['score'].mean():.4f}, Std={target_preds['score'].std():.4f}")
    print(f"  Range: [{target_preds['score'].min():.4f}, {target_preds['score'].max():.4f}]")
    print(f"  >0: {(target_preds['score'] > 0).sum()}/{len(target_preds)}")


if __name__ == "__main__":
    main()
