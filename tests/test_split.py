import numpy as np
import pandas as pd
from src.split import split_dataframe


def test_no_patient_overlap_and_size():
    patients = np.repeat(np.arange(100), 4)                 # 4 rows per patient
    df = pd.DataFrame({"patient_nbr": patients, "target": (patients % 5 == 0).astype(int)})
    train, test = split_dataframe(df, test_size=0.2, seed=42)
    assert set(train["patient_nbr"]).isdisjoint(set(test["patient_nbr"]))
    assert 0.1 < len(test) / len(df) < 0.3
    assert len(train) + len(test) == len(df)