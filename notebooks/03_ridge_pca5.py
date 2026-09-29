"""
=============================================================================
 03_ridge_pca5.py  --  Ridge regression sur 5 composantes PCA
=============================================================================

Identique a 02_ridge_regression.py pour le preprocessing et la CV,
mais les features preprocessees passent par PCA(n_components=5) avant
d'etre donnees au Ridge. L'alpha est choisi par GridSearchCV + GroupKFold.

Sortie : data/y_pred_pca5.csv  (colonnes Index, target)
=============================================================================
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("=" * 60)
print("1. Chargement")
print("=" * 60)

X_train_clean = pd.read_csv("data/X_train_clean.csv")
X_test_clean  = pd.read_csv("data/X_test_clean.csv")
y_train_clean = pd.read_csv("data/y_train_clean.csv")["target"].values

with open("data/feature_config.json") as f:
    feature_config = json.load(f)

NUMERIC_FEATURES     = feature_config["NUMERIC_FEATURES"]
CATEGORICAL_FEATURES = feature_config["CATEGORICAL_FEATURES"]
FLAG_FEATURES        = feature_config["FLAG_FEATURES"]

print(f"Train : {X_train_clean.shape[0]:,} lignes x {X_train_clean.shape[1]} colonnes")
print(f"Test  : {X_test_clean.shape[0]:,} lignes x {X_test_clean.shape[1]} colonnes")

# =============================================================================
# 2. SELECTION DES FEATURES (meme logique que 02)
#    age_at_diagnosis exclue : remplacee par time_since_diagnosis.
#    patient_id extrait pour GroupKFold.
# =============================================================================
numeric_features     = [f for f in NUMERIC_FEATURES if f != "age_at_diagnosis"]
categorical_features = CATEGORICAL_FEATURES
flag_features        = FLAG_FEATURES
all_features         = numeric_features + categorical_features + flag_features

groups  = X_train_clean["patient_id"].values
X_train = X_train_clean[all_features].copy()
X_test  = X_test_clean[all_features].copy()

print(f"\nFeatures utilisees : {len(all_features)}")
print(f"  numeriques+flags : {len(numeric_features) + len(flag_features)}")
print(f"  categorielles    : {len(categorical_features)}")

# =============================================================================
# 3. PIPELINE : preprocesseur -> PCA(5) -> Ridge
#
#    Ordre des etapes :
#    a) ColumnTransformer : imputation+scaling sur num+flags, OHE sur cat
#    b) PCA(n_components=5) : reduit l'espace encode en 5 axes principaux
#    c) Ridge(alpha) : regression lineaire regularisee
#
#    Pourquoi StandardScaler avant PCA ?
#    La PCA est sensible aux echelles. Sans standardisation, les colonnes
#    avec de grandes variances (ex: ledd ~ 1000) domineraient les axes.
# =============================================================================
print("\n" + "=" * 60)
print("3. Construction du pipeline (preprocesseur -> PCA(5) -> Ridge)")
print("=" * 60)

numeric_and_flag = numeric_features + flag_features

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",  StandardScaler()),
            ]),
            numeric_and_flag,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca",          PCA(n_components=5)),
    ("ridge",        Ridge()),
])

print("Etapes : ColumnTransformer -> PCA(n_components=5) -> Ridge")

# =============================================================================
# 4. RECHERCHE D'HYPERPARAMETRES -- GroupKFold par patient
# =============================================================================
print("\n" + "=" * 60)
print("4. GridSearchCV (GroupKFold, n_splits=5)")
print("=" * 60)

param_grid = {"ridge__alpha": np.logspace(-3, 4, 20)}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=GroupKFold(n_splits=5),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

search.fit(X_train, y_train_clean, groups=groups)

best_alpha    = search.best_params_["ridge__alpha"]
best_cv_rmse  = -search.best_score_

print(f"Meilleur alpha : {best_alpha:.4g}")
print(f"RMSE CV (val)  : {best_cv_rmse:.4f}")

# =============================================================================
# 5. EVALUATION CROISEE COMPLETE
# =============================================================================
print("\n" + "=" * 60)
print("5. Evaluation croisee (RMSE / MAE / R2)")
print("=" * 60)

best_pipeline = search.best_estimator_

cv_results = cross_validate(
    best_pipeline,
    X_train,
    y_train_clean,
    cv=GroupKFold(n_splits=5),
    groups=groups,
    scoring={
        "rmse": "neg_root_mean_squared_error",
        "mae":  "neg_mean_absolute_error",
        "r2":   "r2",
    },
    return_train_score=True,
    n_jobs=-1,
)

rmse_val   = -cv_results["test_rmse"].mean()
rmse_std   =  cv_results["test_rmse"].std()
mae_val    = -cv_results["test_mae"].mean()
r2_val     =  cv_results["test_r2"].mean()
rmse_train = -cv_results["train_rmse"].mean()
r2_train   =  cv_results["train_r2"].mean()

print(f"  RMSE  val   : {rmse_val:.4f} +/- {rmse_std:.4f}")
print(f"  MAE   val   : {mae_val:.4f}")
print(f"  R2    val   : {r2_val:.4f}")
print(f"  RMSE  train : {rmse_train:.4f}")
print(f"  R2    train : {r2_train:.4f}")

# Variance expliquee par les 5 composantes (fit sur tout le train)
best_pipeline.fit(X_train, y_train_clean)
pca_fitted = best_pipeline.named_steps["pca"]
var_exp = pca_fitted.explained_variance_ratio_
print(f"\nVariance expliquee par les 5 composantes PCA :")
for i, v in enumerate(var_exp, 1):
    print(f"  PC{i} : {v * 100:.1f}%")
print(f"  Total : {var_exp.sum() * 100:.1f}%")

# =============================================================================
# 6. PREDICTIONS SUR LE TEST
# =============================================================================
print("\n" + "=" * 60)
print("6. Predictions test")
print("=" * 60)

y_pred = best_pipeline.predict(X_test)

print(f"Predictions : min={y_pred.min():.2f}  max={y_pred.max():.2f}"
      f"  mean={y_pred.mean():.2f}  std={y_pred.std():.2f}")

# =============================================================================
# 7. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 60)
print("7. Sauvegarde")
print("=" * 60)

y_pred_df = pd.DataFrame({
    "Index":  X_test_clean["Index"].values,
    "target": y_pred,
})
y_pred_df.to_csv("data/y_pred_pca5.csv", index=False)

print(f"OK data/y_pred_pca5.csv  ({len(y_pred_df):,} lignes)")
print(y_pred_df.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("Termine.")
print("=" * 60)
