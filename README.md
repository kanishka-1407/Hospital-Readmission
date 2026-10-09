## Pipeline

```
data/raw (DVC) → clean → validate → features → split → train / test
```

| Stage | Script | Output |
|---|---|---|
| clean | `src/data_cleaning.py` | `data/processed/clean.parquet` |
| validate | `src/data_validation.py` | `reports/validation_report.json` (quality gate) |
| features | `src/features.py` | `data/processed/features.parquet` |
| split | `src/split.py` | `data/processed/train.parquet`, `test.parquet` |

## Feature engineering
- ICD-9 diagnosis codes grouped into 9 disease categories
- `age` bucket converted to a midpoint (`age_num`); the bucket is kept for the fairness audit
- Admission and discharge ID codes grouped into labelled categories
- New features: `total_prior_visits`, `num_meds_taken`, `num_meds_changed`,
  `lab_tests_per_day`, `meds_per_day`, `a1c_tested`, `glucose_tested`
- Scaling and one-hot encoding are in `src/preprocessing.py` and are fit on the **training set only**
  (no leakage)

## Train/test split
Stratified and grouped by `patient_nbr`, so the same patient never appears in both sets
(`reports/split_report.json` records `patient_overlap: 0`).

## Status
- [x] Repo, DVC and environment setup
- [x] EDA
- [x] Data cleaning, validation and unit tests
- [x] Feature engineering and group-wise split
- [ ] Experiments with MLflow (baseline and LightGBM)
- [ ] FastAPI + Docker
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring (Evidently)