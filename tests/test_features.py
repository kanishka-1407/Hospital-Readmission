import pandas as pd
from src.features import age_midpoint, engineer_features, icd9_group


def test_icd9_groups():
    assert icd9_group("250.83") == "Diabetes"
    assert icd9_group("428") == "Circulatory"
    assert icd9_group("486") == "Respiratory"
    assert icd9_group("V57") == "Other"
    assert icd9_group("Unknown") == "Unknown"


def test_age_midpoint():
    assert age_midpoint("[70-80)") == 75
    assert age_midpoint("[0-10)") == 5


def make_df():
    return pd.DataFrame({
        "age": ["[70-80)", "[50-60)"],
        "diag_1": ["250.1", "428"], "diag_2": ["Unknown", "486"], "diag_3": ["V57", "250"],
        "admission_type_id": ["1", "5"], "admission_source_id": ["7", "99"],
        "discharge_disposition_id": ["1", "2"],
        "number_outpatient": [1, 0], "number_emergency": [0, 2], "number_inpatient": [3, 1],
        "metformin": ["No", "Steady"], "insulin": ["Up", "No"],
        "num_lab_procedures": [40, 10], "num_medications": [10, 4], "time_in_hospital": [4, 2],
        "A1Cresult": ["None", ">8"], "max_glu_serum": ["None", "None"],
    })


def test_engineered_columns():
    out = engineer_features(make_df())
    assert out["total_prior_visits"].tolist() == [4, 3]
    assert out["num_meds_taken"].tolist() == [1, 1]
    assert out["num_meds_changed"].tolist() == [1, 0]
    assert out["lab_tests_per_day"].tolist() == [10.0, 5.0]
    assert out["a1c_tested"].tolist() == [0, 1]
    assert out["admission_source"].tolist() == ["Emergency", "Other"]
    assert "diag_1" not in out.columns and "admission_type_id" not in out.columns
    assert out.isnull().sum().sum() == 0