import pandas as pd
from pandas.api.types import is_string_dtype
from src.data_cleaning import clean_dataframe

PARAMS = {
    "drop_columns": ["weight", "payer_code", "examide", "citoglipton"],
    "expired_hospice_ids": [11, 13, 14, 19, 20, 21],
    "fill_unknown_columns": ["race", "medical_specialty", "diag_1", "diag_2", "diag_3"],
    "id_columns": ["admission_type_id", "discharge_disposition_id", "admission_source_id"],
    "keep_first_encounter_only": False,
}


def make_df():
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5],
        "patient_nbr": [10, 10, 11, 12, 13],
        "gender": ["Male", "Female", "Unknown/Invalid", "Male", "Female"],
        "race": ["Caucasian", None, "Asian", "Asian", "Caucasian"],
        "weight": [None] * 5, "payer_code": [None] * 5,
        "examide": ["No"] * 5, "citoglipton": ["No"] * 5,
        "discharge_disposition_id": [1, 1, 1, 11, 1],
        "admission_type_id": [1] * 5, "admission_source_id": [7] * 5,
        "medical_specialty": [None] * 5,
        "diag_1": ["250"] * 5, "diag_2": ["401"] * 5, "diag_3": [None] * 5,
        "readmitted": ["<30", "NO", ">30", "NO", "NO"],
    })


def test_removes_bad_rows_and_columns():
    out, report = clean_dataframe(make_df(), PARAMS)
    assert len(out) == 3                      # invalid gender + expired removed
    assert "weight" not in out.columns
    assert "readmitted" not in out.columns
    assert report["dropped_invalid_gender"] == 1
    assert report["dropped_expired_hospice"] == 1


def test_target_and_missing_handling():
    out, _ = clean_dataframe(make_df(), PARAMS)
    assert out["target"].tolist() == [1, 0, 0]
    assert out.isnull().sum().sum() == 0
    assert (out["race"] == "Unknown").sum() == 1

def test_id_columns_are_strings():
    out, _ = clean_dataframe(make_df(), PARAMS)
    for col in PARAMS["id_columns"]:
        assert is_string_dtype(out[col])
        assert out[col].iloc[0] == "1" or out[col].iloc[0] == "7"   # values are text, not numbers


def test_keep_first_encounter():
    out, _ = clean_dataframe(make_df(), {**PARAMS, "keep_first_encounter_only": True})
    assert out["patient_nbr"].is_unique