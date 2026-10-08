## Pipeline

```
data/raw (DVC) → clean → data/processed/clean.parquet → validate
```

| Stage | Script | Output |
|---|---|---|
| clean | `src/data_cleaning.py` | `data/processed/clean.parquet`, `reports/cleaning_report.json` |
| validate | `src/data_validation.py` | `reports/validation_report.json` |

Cleaning decisions (see `params.yaml` and `notebooks/01_eda.ipynb`):
- `?` is treated as missing, but `None` in `A1Cresult` and `max_glu_serum` means "test not done"
- Dropped `weight`, `payer_code`, and the constant `examide` and `citoglipton`
- Removed 3 rows with invalid gender and patients who died or went to hospice
- Target: `readmitted == "<30"` → 1 (about 11% positive, so imbalanced)
- `patient_nbr` is kept for a group-wise split to prevent leakage

## How to run

```bash
git clone https://github.com/kanishka-1407/Hospital-Readmission.git
cd Hospital-Readmission
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull          # needs a configured DVC remote
dvc repro
pytest -q
```

## Status
- [x] Repo, DVC and environment setup
- [x] EDA
- [x] Data cleaning pipeline + validation + unit tests
- [ ] Feature engineering
- [ ] Experiments with MLflow
- [ ] FastAPI + Docker
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring (Evidently)