# EDA: Parkinson's Disease Motor Score Dataset

_Generated from `data/eda.py` on 2025-07-10._

## Dataset at a glance

- **Tables:** `X_train.csv` (features), `y_train.csv` (target), `X_test.csv` (features for submission)
- **Shape:**
  - Train: 44 590 rows × 12 feature columns + 1 target column
  - Test: 11 013 rows × 12 feature columns (no target)
- **Target:** `target` — debiased true OFF MDS-UPDRS motor score (continuous, 0–132) — **regression**
- **Groups:** 5 576 unique patients in train, 1 395 in test; **zero patient overlap** between train and test
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | Type | Missing (train) | Missing (test) | Notes |
|---|---|---|---|---|
| `Index` | int | 0% | 0% | Row identifier, not a feature |
| `patient_id` | string | 0% | 0% | Patient identifier; ~8 visits/patient on average |
| `cohort` | string | 0% | 0% | Two cohorts: `A` (88.9%) and `B` (11.1%) |
| `sexM` | binary int | 0% | 0% | 60% male (1), 40% female (0) |
| `gene` | string | **32.4%** | 32.0% | Categories: `No Mutation`, `LRRK2+`, `GBA+`, `OTHER+`; ~1/3 missing |
| `age_at_diagnosis` | float | **5.2%** | 4.6% | Age at PD diagnosis; needed to compute disease duration |
| `age` | float | 0% | 0% | Current age at visit; always present |
| `ledd` | float | **36.6%** | 38.6% | Levodopa-equivalent daily dose; missing likely means no DRT yet |
| `time_since_intake_on` | float | **46.4%** | 47.8% | Hours since last intake at time of ON exam |
| `time_since_intake_off` | float | **78.8%** | 79.1% | Hours since last intake at time of OFF exam — extremely sparse |
| `on` | float | **29.6%** | 31.2% | Observed ON motor score (biased) |
| `off` | float | **42.4%** | 40.8% | Observed OFF motor score (biased); absent when OFF exam was skipped |

**Key observations:**
- `time_since_intake_off` is the most missing column (79%) — OFF exams are rare and uncomfortable; their absence is informative.
- `on` and `off` are the biased clinical scores; the task is to predict the debiased version (`target`). They are strong predictors but **not leaks** — the model is expected to debias them using drug timing and progression context.
- `ledd` missing likely indicates the patient was not yet on dopamine replacement therapy at that visit.
- `gene` has exactly 4 non-null categories; the 32% missingness is a structural feature of multi-cohort data collection, not noise.

## Target

The target is the **debiased true OFF MDS-UPDRS motor score**, a continuous value between 0 and 132.

| Statistic | Value |
|---|---|
| Count | 44 590 |
| Mean | 37.47 |
| Std | 16.50 |
| Min | 0.00 |
| p25 | 25.60 |
| Median | 37.30 |
| p75 | 49.30 |
| Max | 109.50 |

The distribution is approximately symmetric around the mean (~37), with a standard deviation of ~16.5. No extreme skew — a standard RMSE objective is appropriate without a target transform. The range is wide (0 to 109.5), reflecting the full spectrum from early-stage to advanced PD patients.

## Structure

- **No datetime columns** — visits are identified by `age` (current age at visit) and an implicit temporal order within each patient's `patient_id` group.
- **Group structure (critical):** `patient_id` has 5 576 unique values across 44 590 rows (unique ratio ≈ 0.125). Each patient contributes ~8 visits on average. The Kaggle holdout is **by patient** (zero overlap confirmed). Shuffling rows in CV will leak visits from the same patient into both train and validation — this is the primary CV risk.
- **No explicit timestamp column** — but `age` is a monotonically increasing proxy for time within a patient. Cross-sectional analysis across patients is the primary task; intra-patient temporal progression is a secondary signal to capture.

## Associations

Strong pairwise associations found (from direct analysis):

| Pair | Pearson r | Interpretation |
|---|---|---|
| `off` ↔ `target` | **0.886** | Biased OFF score is the strongest single predictor of the true OFF |
| `on` ↔ `target` | **0.688** | ON score also highly predictive (reflects underlying disease burden) |
| `age` ↔ `target` | positive | Older patients tend to have higher motor burden |
| `ledd` ↔ `target` | positive | Higher dose suggests more advanced disease |

**No leakage risk identified.** The `off` and `on` scores are the biased clinical measurements the model is explicitly supposed to use and debias — their high correlation with `target` is by design, not an artefact.

`patient_id` and `Index` are identifiers. If left in features, the model could memorize patient-level effects. **Drop both before fitting.**

## Modelling implications

- **`GroupKFold` on `patient_id` is mandatory.** The Kaggle holdout is by patient; any row-level shuffle leaks same-patient visits across train/val. Use `GroupKFold(n_splits=5)` with `groups = patient_id`. This is the single highest-priority data decision.
- **Missingness is signal, not noise.** `time_since_intake_off` (79% missing) and `off` (42% missing) encode visit type: a missing OFF exam typically means the visit was ON-only, which is clinically meaningful. Use `HistGradientBoostingRegressor` (native NaN support) rather than imputing.
- **`gene` and `cohort` are categorical features.** Include them via `skrub.TableVectorizer` or manual one-hot encoding. `gene` has moderate cardinality (4 categories + missing); `cohort` has 2. These are informative and should not be dropped.
- **Drop `Index` and `patient_id` from features.** They are identifiers that would cause leakage by patient memorisation.
- **Disease duration as a derived feature.** `age - age_at_diagnosis` gives years since diagnosis when `age_at_diagnosis` is present (~95% of rows). This encodes disease progression, a key mechanistic variable.
- **Drug timing features (`time_since_intake_on`, `time_since_intake_off`, `ledd`).** These are the levodopa pharmacodynamic signal. Their missingness pattern (no ledd → no DRT yet) is informative. Keep as-is for tree-based models.
- **Target is continuous, approximately symmetric.** Standard RMSE metric. No target transform required initially.
- **Cohort imbalance (A: 89%, B: 11%)** — may create subgroup performance differences. Worth checking per-cohort RMSE after training.

## Open questions

1. **`gene` missingness mechanism:** Is it missing-at-random (not collected in some cohorts) or missing-informatively (certain patient profiles not genotyped)? The 32% rate is consistent across train and test, suggesting a structural collection gap rather than a modelling gap.
2. **Multiple visits per patient:** Should intra-patient temporal trends (score worsening over visits) be explicitly modelled? A recurrent or lag-feature approach might improve over treating visits as i.i.d.
3. **`off` used as feature:** The model is debiasing the observed `off` score. Is it valid to include `off` as a feature when it is 42% missing? Yes — the model should learn from both the value when present AND the absence pattern.
4. **Cohort B:** Only 11.1% of train. Are cohort-specific effects large enough to warrant stratification by cohort in the CV splits?
