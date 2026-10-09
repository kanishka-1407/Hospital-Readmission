import json
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_PATH = Path("reports/features_report.json")

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Verify these groupings against your IDs_mapping.csv
ADMISSION_TYPE_MAP = {
    "1": "Emergency", "2": "Urgent", "3": "Elective",
    "5": "Unknown", "6": "Unknown", "8": "Unknown",
}
ADMISSION_SOURCE_MAP = {
    "7": "Emergency", "1": "Referral", "2": "Referral", "3": "Referral",
    "4": "Transfer", "5": "Transfer", "6": "Transfer", "10": "Transfer",
    "22": "Transfer", "25": "Transfer", "26": "Transfer",
    "9": "Unknown", "15": "Unknown", "17": "Unknown", "20": "Unknown", "21": "Unknown",
}
DISCHARGE_MAP = {
    "1": "Home", "6": "HomeHealth", "8": "HomeHealth",
    "2": "Facility", "3": "Facility", "4": "Facility", "5": "Facility",
    "22": "Facility", "23": "Facility", "24": "Facility", "27": "Facility",
    "28": "Facility", "29": "Facility", "30": "Facility",
    "18": "Unknown", "25": "Unknown", "26": "Unknown",
}


def icd9_group(code) -> str:
    """Group an ICD-9 code into broad disease categories."""
    code = str(code)
    if code == "Unknown":
        return "Unknown"
    if code.startswith(("V", "E")):
        return "Other"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    if 250 <= n < 251:
        return "Diabetes"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def age_midpoint(bucket: str) -> int:
    """'[70-80)' -> 75"""
    return int(bucket.strip("[)").split("-")[0]) + 5


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure row-wise transformations (safe before the train/test split)."""
    df = df.copy()

    # 1. ICD-9 codes -> disease groups
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].map(icd9_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # 2. Age bucket -> number (the bucket column stays for the fairness audit later)
    df["age_num"] = df["age"].map(age_midpoint)

    # 3. Group the coded ID columns into readable categories
    df["admission_type"] = df["admission_type_id"].map(lambda x: ADMISSION_TYPE_MAP.get(x, "Other"))
    df["admission_source"] = df["admission_source_id"].map(lambda x: ADMISSION_SOURCE_MAP.get(x, "Other"))
    df["discharge_group"] = df["discharge_disposition_id"].map(lambda x: DISCHARGE_MAP.get(x, "Other"))
    df = df.drop(columns=["admission_type_id", "admission_source_id", "discharge_disposition_id"])

    # 4. Prior-visit history
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # 5. Medication summaries
    drugs = [c for c in DRUG_COLUMNS if c in df.columns]
    df["num_meds_taken"] = (df[drugs] != "No").sum(axis=1)
    df["num_meds_changed"] = df[drugs].isin(["Up", "Down"]).sum(axis=1)

    # 6. Intensity ratios (time_in_hospital >= 1 is checked by the validation stage)
    df["lab_tests_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"]
    df["meds_per_day"] = df["num_medications"] / df["time_in_hospital"]

    # 7. Was the test performed? ("None" means not done, which is informative)
    df["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)
    df["glucose_tested"] = (df["max_glu_serum"] != "None").astype(int)

    return df


def main():
    df = pd.read_parquet(CLEAN_PATH)
    out = engineer_features(df)
    assert out.isnull().sum().sum() == 0, "NaNs introduced during feature engineering"

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH, index=False)

    report = {
        "rows": len(out),
        "columns": out.shape[1],
        "new_features": [
            "diag_1_group", "diag_2_group", "diag_3_group", "age_num", "admission_type",
            "admission_source", "discharge_group", "total_prior_visits", "num_meds_taken",
            "num_meds_changed", "lab_tests_per_day", "meds_per_day", "a1c_tested", "glucose_tested",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["rows", "columns"]}))


if __name__ == "__main__":
    main()