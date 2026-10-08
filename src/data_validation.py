import json
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

CLEAN_PATH = Path("data/processed/clean.parquet")
REPORT_PATH = Path("reports/validation_report.json")

AGE_BUCKETS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
REMOVED_IDS = ["11", "13", "14", "19", "20", "21"]

schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(int, unique=True),
        "patient_nbr": pa.Column(int),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "age": pa.Column(str, pa.Check.isin(AGE_BUCKETS)),
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),
        "num_medications": pa.Column(int, pa.Check.ge(0)),
        "number_inpatient": pa.Column(int, pa.Check.ge(0)),
        "discharge_disposition_id": pa.Column(
            str, pa.Check(lambda s: ~s.isin(REMOVED_IDS), error="expired/hospice ids present")
        ),
        "target": pa.Column(int, pa.Check.isin([0, 1])),
    },
    checks=[
        pa.Check(lambda d: d.notna().all().all(), error="null values present"),
        pa.Check(lambda d: 0.05 < d["target"].mean() < 0.20, error="target rate outside 5-20%"),
    ],
    strict=False,
    coerce=True,
)


def main():
    df = pd.read_parquet(CLEAN_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        schema.validate(df, lazy=True)
        report = {"status": "passed", "rows": len(df), "failed_checks": 0}
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases.head(20))
        report = {"status": "failed", "rows": len(df), "failed_checks": int(len(e.failure_cases))}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(report)
    if report["status"] == "failed":
        sys.exit(1)      # fails the DVC stage, and later the CI pipeline


if __name__ == "__main__":
    main()