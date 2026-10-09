import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Never used as model inputs. `age` stays only for the fairness audit.
NON_FEATURE_COLUMNS = ["encounter_id", "patient_nbr", "target", "age"]


def get_feature_columns(df: pd.DataFrame):
    X = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()   # works with pandas 3 str dtype
    return numeric, categorical


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    """Unfitted. Call .fit() on the TRAIN set only (done in the training step)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",   # unseen categories do not crash the API
                    min_frequency=0.01,                      # rare categories are grouped as "infrequent"
                    sparse_output=False,
                ),
                categorical,
            ),
        ]
    )