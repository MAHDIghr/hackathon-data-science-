"""
=============================================================================
 02_ridge_regression.py  —  Régression Ridge avec CV groupée par patient
=============================================================================

MODÈLE :
  - Feature engineering depuis feature_config.json (nettoyage 01_data_cleaning)
  - Ridge avec sélection d'alpha par GridSearchCV + GroupKFold(patient_id)
  - Métriques de pertinence : RMSE, MAE, R² (CV et train complet)
  - Sortie : data/y_pred.csv (même format que y_train.csv)

NOTE SUR age_at_diagnosis :
  La colonne age_at_diagnosis est EXCLUE des features directes.
  Elle sert uniquement à calculer time_since_diagnosis = age - age_at_diagnosis,
  déjà produite par clean_and_engineer() dans 01_data_cleaning.py.
=============================================================================
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES NETTOYÉES + CONFIG FEATURES
# =============================================================================
print("=" * 60)
print("1. Chargement")
print("=" * 60)

X_train_clean = pd.read_csv("data/X_train_clean.csv")
X_test_clean  = pd.read_csv("data/X_test_clean.csv")
y_train_clean = pd.read_csv("data/y_train_clean.csv")["target"].values

with open("data/feature_config.json") as f:
    feature_config = json.load(f)

NUMERIC_FEATURES      = feature_config["NUMERIC_FEATURES"]
CATEGORICAL_FEATURES  = feature_config["CATEGORICAL_FEATURES"]
FLAG_FEATURES         = feature_config["FLAG_FEATURES"]

print(f"Train : {X_train_clean.shape[0]:,} lignes × {X_train_clean.shape[1]} colonnes")
print(f"Test  : {X_test_clean.shape[0]:,} lignes × {X_test_clean.shape[1]} colonnes")


# =============================================================================
# 2. SÉLECTION DES FEATURES
#    age_at_diagnosis est exclue : time_since_diagnosis la remplace.
#    patient_id est extrait séparément pour le GroupKFold.
# =============================================================================
print("\n" + "=" * 60)
print("2. Sélection des features")
print("=" * 60)

# Retirer age_at_diagnosis des numériques (remplacée par time_since_diagnosis)
numeric_features = [f for f in NUMERIC_FEATURES if f != "age_at_diagnosis"]
categorical_features = CATEGORICAL_FEATURES
flag_features = FLAG_FEATURES

all_features = numeric_features + categorical_features + flag_features

print(f"Features numériques ({len(numeric_features)}) : {numeric_features}")
print(f"Features catégorielles ({len(categorical_features)}) : {categorical_features}")
print(f"Features binaires/flags ({len(flag_features)}) : {flag_features}")
print(f"Total : {len(all_features)} features")

# Extraire les groupes (patient_id) avant de restreindre les colonnes
groups = X_train_clean["patient_id"].values

X_train = X_train_clean[all_features].copy()
X_test  = X_test_clean[all_features].copy()


# =============================================================================
# 3. PIPELINE SKLEARN
#    - Numériques + flags : imputation mean → standardisation
#    - Catégorielles : OHE (handle_unknown=ignore pour le test set)
# =============================================================================
print("\n" + "=" * 60)
print("3. Construction du pipeline")
print("=" * 60)

numeric_and_flag = numeric_features + flag_features  # traitement identique

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
    ("ridge",        Ridge()),
])

print("Pipeline :")
print("  num (imputer mean -> StandardScaler) :", numeric_and_flag)
print("  cat (OneHotEncoder) :", categorical_features)
print("  estimateur : Ridge(alpha=?)")


# =============================================================================
# 4. RECHERCHE D'HYPERPARAMÈTRES — GroupKFold par patient
#    Alpha balaye 4 décades (10⁻³ → 10⁴) en 20 valeurs log-espacées.
#    GroupKFold(5) garantit qu'aucun patient n'est à la fois train et val.
# =============================================================================
print("\n" + "=" * 60)
print("4. GridSearchCV (GroupKFold, n_splits=5)")
print("=" * 60)

alphas = np.logspace(-3, 4, 20)
param_grid = {"ridge__alpha": alphas}

cv_strategy = GroupKFold(n_splits=5)

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_strategy,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)

search.fit(X_train, y_train_clean, groups=groups)

best_alpha = search.best_params_["ridge__alpha"]
best_cv_rmse = -search.best_score_

print(f"Meilleur alpha : {best_alpha:.4g}")
print(f"RMSE CV (val)  : {best_cv_rmse:.4f}")


# =============================================================================
# 5. PERTINENCE DU MODÈLE — cross_validate complet (RMSE, MAE, R²)
#    On re-fait une CV sur le pipeline best_alpha pour obtenir MAE et R².
# =============================================================================
print("\n" + "=" * 60)
print("5. Évaluation croisée complète (RMSE / MAE / R²)")
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
rmse_std   = cv_results["test_rmse"].std()
mae_val    = -cv_results["test_mae"].mean()
mae_std    = cv_results["test_mae"].std()
r2_val     = cv_results["test_r2"].mean()
r2_std     = cv_results["test_r2"].std()
rmse_train = -cv_results["train_rmse"].mean()
r2_train   = cv_results["train_r2"].mean()

print(f"  RMSE  val   : {rmse_val:.4f} +/- {rmse_std:.4f}")
print(f"  MAE   val   : {mae_val:.4f} +/- {mae_std:.4f}")
print(f"  R2    val   : {r2_val:.4f} +/- {r2_std:.4f}")
print(f"  RMSE  train : {rmse_train:.4f}  (surapprentissage check)")
print(f"  R2    train : {r2_train:.4f}")

gap_rmse = rmse_train / rmse_val
print(f"\n  Ratio RMSE train/val : {gap_rmse:.3f}  (< 0.8 -> fort surapprentissage)")
if r2_val < 0.3:
    print("  [!] R2 < 0.3 : pouvoir predictif faible avec Ridge lineaire.")
elif r2_val < 0.6:
    print("  [~] R2 modere : Ridge capte une partie du signal.")
else:
    print("  [ok] R2 > 0.6 : bon pouvoir predictif.")


# =============================================================================
# 6. RE-FIT SUR TOUT LE TRAIN + PRÉDICTION TEST
# =============================================================================
print("\n" + "=" * 60)
print("6. Fit final + prédictions")
print("=" * 60)

best_pipeline.fit(X_train, y_train_clean)
y_pred = best_pipeline.predict(X_test)

print(f"Prédictions : min={y_pred.min():.2f}  max={y_pred.max():.2f}"
      f"  mean={y_pred.mean():.2f}  std={y_pred.std():.2f}")


# =============================================================================
# 7. SAUVEGARDE — même format que y_train.csv (colonnes Index, target)
# =============================================================================
print("\n" + "=" * 60)
print("7. Sauvegarde")
print("=" * 60)

y_pred_df = pd.DataFrame({
    "Index":  X_test_clean["Index"].values,
    "target": y_pred,
})
y_pred_df.to_csv("data/y_pred.csv", index=False)

print(f"OK data/y_pred.csv  ({len(y_pred_df):,} lignes)")
print(f"   Colonnes : {list(y_pred_df.columns)}")
print("\nAperçu :")
print(y_pred_df.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("Terminé.")
print("=" * 60)
