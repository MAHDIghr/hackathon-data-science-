"""
=============================================================================
 03_hgbr.py  —  LightGBM (Histogram Gradient Boosting) + features temporelles
=============================================================================

MODÈLE :
  - LightGBMRegressor — même algorithme que HistGradientBoostingRegressor
    (gradient boosting sur histogrammes), mais sans les DLLs sklearn.tree
  - Features issues de feature_config.json + 4 nouvelles features temporelles :
      prev_off_2   : lag 2 de off par patient
      delta_off    : off(t) - off(t-1)  (vitesse de dégradation motrice)
      delta_on     : on(t)  - on(t-1)   (évolution de la réponse médicament)
      intake_ratio : time_since_intake_on / time_since_intake_off
  - Catégorielles déclarées à LightGBM (gérées nativement)
  - Pas d'imputation — LightGBM gère les NaN nativement (signal préservé)
  - GroupKFold(5) par patient_id — aucun patient commun entre train et val
  - Métriques : RMSE, MAE, R² (CV val + train)
  - Sortie : data/y_pred_hgbr.csv (colonnes Index, target)
=============================================================================
"""

import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, cross_validate

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

NUMERIC_FEATURES     = feature_config["NUMERIC_FEATURES"]
CATEGORICAL_FEATURES = feature_config["CATEGORICAL_FEATURES"]
FLAG_FEATURES        = feature_config["FLAG_FEATURES"]

print(f"Train : {X_train_clean.shape[0]:,} lignes × {X_train_clean.shape[1]} colonnes")
print(f"Test  : {X_test_clean.shape[0]:,} lignes × {X_test_clean.shape[1]} colonnes")


# =============================================================================
# 2. NOUVELLES FEATURES TEMPORELLES
#    Calculées par patient trié par âge, puis réaligné sur l'Index original.
#    Les NaN sont volontaires — LightGBM les exploite comme signal.
# =============================================================================
print("\n" + "=" * 60)
print("2. Ajout des features temporelles")
print("=" * 60)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 4 features temporelles intra-patient :
      - prev_off_2   : lag 2 de off (progression sur 2 visites)
      - delta_off    : off(t) - off(t-1) (vitesse de dégradation motrice)
      - delta_on     : on(t)  - on(t-1)  (évolution de la réponse médicament)
      - intake_ratio : time_since_intake_on / time_since_intake_off
    """
    df = df.copy()
    df = df.sort_values(["patient_id", "age"]).reset_index(drop=True)

    df["prev_off_2"]   = df.groupby("patient_id")["off"].shift(2)
    df["delta_off"]    = df["off"] - df.groupby("patient_id")["off"].shift(1)
    df["delta_on"]     = df["on"]  - df.groupby("patient_id")["on"].shift(1)
    df["intake_ratio"] = df["time_since_intake_on"] / df["time_since_intake_off"]

    df = df.sort_values("Index").reset_index(drop=True)
    return df


X_train_clean = add_temporal_features(X_train_clean)
X_test_clean  = add_temporal_features(X_test_clean)

NEW_FEATURES = ["prev_off_2", "delta_off", "delta_on", "intake_ratio"]
print(f"Nouvelles features : {NEW_FEATURES}")
for f in NEW_FEATURES:
    n_nan = X_train_clean[f].isnull().sum()
    pct   = 100 * n_nan / len(X_train_clean)
    print(f"  {f:20s}  NaN={n_nan:,}  ({pct:.1f}%)")


# =============================================================================
# 3. SÉLECTION DES FEATURES
#    age_at_diagnosis exclue (remplacée par time_since_diagnosis).
#    patient_id extrait pour GroupKFold.
#    Catégorielles converties en type "category" pour LightGBM.
# =============================================================================
print("\n" + "=" * 60)
print("3. Sélection des features")
print("=" * 60)

numeric_features     = [f for f in NUMERIC_FEATURES if f != "age_at_diagnosis"]
categorical_features = CATEGORICAL_FEATURES
flag_features        = FLAG_FEATURES

all_features = numeric_features + NEW_FEATURES + categorical_features + flag_features

groups  = X_train_clean["patient_id"].values
X_train = X_train_clean[all_features].copy()
X_test  = X_test_clean[all_features].copy()

# LightGBM détecte les catégorielles via le dtype "category"
for col in categorical_features:
    X_train[col] = X_train[col].astype("category")
    X_test[col]  = X_test[col].astype("category")

print(f"Numériques ({len(numeric_features)})    : {numeric_features}")
print(f"Temporelles ({len(NEW_FEATURES)})   : {NEW_FEATURES}")
print(f"Catégorielles ({len(categorical_features)})  : {categorical_features}")
print(f"Flags ({len(flag_features)})         : {flag_features}")
print(f"Total : {len(all_features)} features")


# =============================================================================
# 4. MODÈLE LIGHTGBM
#    - verbose=-1 : silence les logs d'entraînement
#    - n_jobs=-1  : tous les cœurs CPU
#    - Paramètres calqués sur les recommandations HGBR du guide
# =============================================================================
print("\n" + "=" * 60)
print("4. Configuration du modèle")
print("=" * 60)

model = LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=20,
    random_state=0,
    verbose=-1,
    n_jobs=-1,
)

print("Modèle : LGBMRegressor(")
print("  n_estimators=500, max_depth=6, learning_rate=0.05,")
print("  min_child_samples=20, random_state=0)")


# =============================================================================
# 5. ÉVALUATION CROISÉE — GroupKFold par patient (RMSE / MAE / R²)
# =============================================================================
print("\n" + "=" * 60)
print("5. Évaluation croisée (GroupKFold, n_splits=5)")
print("=" * 60)

cv_results = cross_validate(
    model,
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
mae_std    =  cv_results["test_mae"].std()
r2_val     =  cv_results["test_r2"].mean()
r2_std     =  cv_results["test_r2"].std()
rmse_train = -cv_results["train_rmse"].mean()
r2_train   =  cv_results["train_r2"].mean()

print(f"  RMSE  val   : {rmse_val:.4f} +/- {rmse_std:.4f}")
print(f"  MAE   val   : {mae_val:.4f} +/- {mae_std:.4f}")
print(f"  R2    val   : {r2_val:.4f} +/- {r2_std:.4f}")
print(f"  RMSE  train : {rmse_train:.4f}  (surapprentissage check)")
print(f"  R2    train : {r2_train:.4f}")

gap_rmse = rmse_train / rmse_val
print(f"\n  Ratio RMSE train/val : {gap_rmse:.3f}  (< 0.8 -> surapprentissage)")
if r2_val < 0.3:
    print("  [!] R2 < 0.3 : pouvoir prédictif faible.")
elif r2_val < 0.6:
    print("  [~] R2 modéré : modèle capte une partie du signal.")
else:
    print("  [ok] R2 > 0.6 : bon pouvoir prédictif.")

print(f"\n  Comparaison avec Ridge (02_ridge_regression.py) :")
print(f"    Ridge RMSE : 7.4255  ->  LightGBM RMSE : {rmse_val:.4f}"
      f"  (gain : {7.4255 - rmse_val:+.4f})")
print(f"    Ridge R2   : 0.7972  ->  LightGBM R2   : {r2_val:.4f}"
      f"  (gain : {r2_val - 0.7972:+.4f})")


# =============================================================================
# 6. RE-FIT SUR TOUT LE TRAIN + PRÉDICTION TEST
# =============================================================================
print("\n" + "=" * 60)
print("6. Fit final + prédictions")
print("=" * 60)

model.fit(X_train, y_train_clean)
y_pred = model.predict(X_test)

print(f"Prédictions : min={y_pred.min():.2f}  max={y_pred.max():.2f}"
      f"  mean={y_pred.mean():.2f}  std={y_pred.std():.2f}")


# =============================================================================
# 7. SAUVEGARDE — même format que sample_submission.csv (colonnes Index, target)
# =============================================================================
print("\n" + "=" * 60)
print("7. Sauvegarde")
print("=" * 60)

y_pred_df = pd.DataFrame({
    "Index":  X_test_clean["Index"].values,
    "target": y_pred,
})
y_pred_df.to_csv("data/y_pred_hgbr.csv", index=False)

print(f"OK data/y_pred_hgbr.csv  ({len(y_pred_df):,} lignes)")
print(f"   Colonnes : {list(y_pred_df.columns)}")
print("\nAperçu :")
print(y_pred_df.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("Terminé.")
print("=" * 60)
