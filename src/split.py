import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

FEATURES_PATH = Path("data/processed/features.parquet")
TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")
REPORT_PATH = Path("reports/split_report.json")


def split_dataframe(df: pd.DataFrame, test_size: float, seed: int):
    """Same patient never appears in both sets; target ratio is kept similar."""
    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["target"], groups=df["patient_nbr"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["split"]

    df = pd.read_parquet(FEATURES_PATH)
    train, test = split_dataframe(df, p["test_size"], p["random_state"])

    overlap = len(set(train["patient_nbr"]) & set(test["patient_nbr"]))
    assert overlap == 0, f"{overlap} patients leaked across train/test"

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_fraction": round(len(test) / len(df), 4),
        "train_target_rate": round(float(train["target"].mean()), 4),
        "test_target_rate": round(float(test["target"].mean()), 4),
        "patient_overlap": overlap,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()