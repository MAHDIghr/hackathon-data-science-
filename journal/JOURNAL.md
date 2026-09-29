# Journal — Parkinson's Motor Score (IBM/Probabl Hackathon)

**Goal:** Predict the debiased true OFF MDS-UPDRS motor score (`target`) for each
patient visit. Regression task; public Kaggle metric is RMSE.
Every submission requires a Skore Hub EstimatorReport URL in the Kaggle
Submission Description.

## Status

| Decision | Choice | Rationale |
|---|---|---|
| Tabular library | pandas | Default stack; CSVs loaded with `pd.read_csv` |
| CV splitter | `GroupKFold(n_splits=5)` on `patient_id` | Kaggle holdout is by patient; row shuffle leaks visits |
| Primary metric | RMSE | Continuous target, approximately symmetric |
| Agent feature | blocked by OS DLL policy (pandas Cython DLLs) | Skore runner cannot execute; EDA authored from direct CSV analysis |

## Data understanding (EDA)

- **Status:** done - 2025-07-10
- **Summary:** 44 590 train rows × 12 features across 5 576 patients (avg 8 visits/patient); zero patient overlap with 11 013-row test set. Target (`target`, debiased true OFF score) is continuous, mean ≈ 37.5, std ≈ 16.5 — no extreme skew. Heavy missingness is structural signal (`time_since_intake_off` 79% missing, `off` 42% missing) — use `HistGradientBoostingRegressor` with native NaN support. Patient grouping makes `GroupKFold` mandatory to avoid leakage.
- **Report:** [data/eda.md](../data/eda.md)

## Experiment log

_No experiments run yet._

| # | Key | Method | CV RMSE | Notes |
|---|---|---|---|---|
