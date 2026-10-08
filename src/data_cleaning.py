import json
from pathlib import Path

import pandas as pd
import yaml

RAW_PATH = Path("data/raw/diabetic_data.csv")
CLEAN_PATH = Path("data/processed/clean.parquet")
REPORT_PATH = Path("reports/cleaning_report.json")


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["clean"]


def extract(path: Path = RAW_PATH) -> pd.DataFrame:
    # "?" is missing. keep_default_na=False so the text "None" stays a real category.
    return pd.read_csv(path, na_values=["?"], keep_default_na=False)


def clean_dataframe(df: pd.DataFrame, p: dict):
    """Transform: pure function (no file I/O), so it is easy to unit test."""
    report = {"rows_raw": len(df), "cols_raw": df.shape[1]}
    df = df.copy()

    # 1. Fully duplicate rows
    n = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicate_rows"] = n - len(df)

    # 2. Drop useless columns (mostly empty or constant)
    df = df.drop(columns=p["drop_columns"], errors="ignore")

    # 3. Invalid gender
    n = len(df)
    df = df[df["gender"].isin(["Male", "Female"])]
    report["dropped_invalid_gender"] = n - len(df)

    # 4. Patients who died or went to hospice cannot be readmitted
    n = len(df)
    df = df[~df["discharge_disposition_id"].isin(p["expired_hospice_ids"])]
    report["dropped_expired_hospice"] = n - len(df)

    # 5. Optional: keep only the first encounter per patient
    if p["keep_first_encounter_only"]:
        n = len(df)
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
        report["dropped_repeat_patients"] = n - len(df)

    # 6. Real missing values -> explicit "Unknown" category
    #    (a constant fill is not learned from data, so it is safe before the split)
    for col in p["fill_unknown_columns"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 7. Binary target: readmitted within 30 days = 1. Drop the original to avoid leakage.
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 8. ID columns are categories, not numbers
    for col in p["id_columns"]:
        df[col] = df[col].astype(str)

    df = df.reset_index(drop=True)

    report["rows_clean"] = len(df)
    report["cols_clean"] = df.shape[1]
    report["unique_patients"] = int(df["patient_nbr"].nunique())
    report["target_rate"] = round(float(df["target"].mean()), 4)
    report["remaining_nulls"] = int(df.isnull().sum().sum())
    return df, report


def main():
    params = load_params()
    df = extract()
    df, report = clean_dataframe(df, params)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)          # Load
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()