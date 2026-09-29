"""
=============================================================================
 04_regularized_enhanced.py  --  L2 / L1 / L1+L2 avec features enrichies
=============================================================================

Améliorations par rapport à 02_ridge_regression.py :

  1. FEATURES ENRICHIES
     - levo_concentration  : proxy de la concentration de lévodopa au moment
                             de l'examen (modèle pharmacocinétique simplifié)
                             = ledd * exp(-time_since_intake_on / 2.0)
     - mean_past_off       : moyenne des scores OFF des visites précédentes
                             du même patient (mémoire de la trajectoire)
     - mean_past_on        : idem pour ON
     - off_on_ratio        : rapport OFF/ON quand les deux sont disponibles

  2. TROIS RÉGULARISATIONS COMPARÉES
     - Ridge    (L2) : pénalise la somme des carrés des coefficients
                       → tous les coefficients restent petits mais non nuls
     - Lasso    (L1) : pénalise la somme des valeurs absolues
                       → sélection automatique de features (coefficients = 0)
     - ElasticNet (L1+L2) : combinaison des deux
                       → robuste quand des features sont corrélées entre elles

  3. SÉLECTION D'HYPERPARAMÈTRES
     GridSearchCV + GroupKFold(5) par patient_id pour chaque modèle.
     Alpha balaye 10^-3 à 10^6 (plage élargie : Ridge a convergé à 144 en 02,
     Lasso a besoin d'une plage plus large car l'échelle de la pénalité diffère).

  4. SORTIE
     data/y_pred_best.csv  : prédictions du meilleur modèle (lowest RMSE CV)
     Tableau comparatif RMSE imprimé en fin de script.
=============================================================================
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
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
# 2. FEATURES ENRICHIES
#    Toutes les nouvelles colonnes sont calculées SÉPARÉMENT sur train et test
#    pour éviter toute fuite d'information.
#
#    Les statistiques patient (mean_past_off, mean_past_on) utilisent uniquement
#    les visites PRÉCÉDENTES (shift+expanding) : pas de leakage temporel.
# =============================================================================
print("\n" + "=" * 60)
print("2. Feature engineering enrichi")
print("=" * 60)


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features pharmacocinétiques et d'historique patient.
    df doit être trié par (patient_id, age) -- déjà fait par clean_and_engineer.
    """
    df = df.copy()

    # --- Feature pharmacocinétique ----------------------------------------
    # Proxy de la concentration de lévodopa au moment de l'examen.
    # Modèle simplifié : C(t) ~ dose * exp(-t / demi_vie)
    # On utilise time_since_intake_on comme variable de temps post-dose.
    # Quand time_since_intake_on est NaN (visite OFF-only), la concentration
    # est NaN -- l'imputer la gèrera ensuite.
    HALF_LIFE = 2.0  # heures, demi-vie approximative de la lévodopa
    df["levo_concentration"] = (
        df["ledd"] * np.exp(-df["time_since_intake_on"] / HALF_LIFE)
    )

    # --- Rapport OFF/ON ----------------------------------------------------
    # Quand les deux scores sont disponibles, leur rapport mesure l'efficacité
    # du traitement. NaN sinon (géré par l'imputer).
    df["off_on_ratio"] = np.where(
        df["on"].notna() & df["off"].notna() & (df["on"] > 0),
        df["off"] / df["on"],
        np.nan,
    )

    # --- Historique patient (moyenne des visites passées) ------------------
    # Trié par patient_id + age pour que shift(1) = visite précédente.
    df = df.sort_values(["patient_id", "age"]).reset_index(drop=True)

    df["mean_past_off"] = (
        df.groupby("patient_id")["off"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["mean_past_on"] = (
        df.groupby("patient_id")["on"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Réordonner selon Index original pour l'alignement avec y
    df = df.sort_values("Index").reset_index(drop=True)
    return df


X_train_enh = add_enhanced_features(X_train_clean)
X_test_enh  = add_enhanced_features(X_test_clean)

new_features = ["levo_concentration", "off_on_ratio", "mean_past_off", "mean_past_on"]
for f in new_features:
    n_nan_tr = X_train_enh[f].isnull().sum()
    n_nan_te = X_test_enh[f].isnull().sum()
    print(f"  {f:25s}  NaN train={n_nan_tr:,}  NaN test={n_nan_te:,}")


# =============================================================================
# 3. LISTE DE FEATURES FINALE
# =============================================================================
numeric_features     = [f for f in NUMERIC_FEATURES if f != "age_at_diagnosis"]
categorical_features = CATEGORICAL_FEATURES
flag_features        = FLAG_FEATURES
extra_features       = new_features   # les 4 nouvelles

all_features = numeric_features + categorical_features + flag_features + extra_features

groups  = X_train_enh["patient_id"].values
X_train = X_train_enh[all_features].copy()
X_test  = X_test_enh[all_features].copy()

print(f"\nTotal features : {len(all_features)}")
print(f"  (base: {len(numeric_features)+len(flag_features)} num+flags,"
      f"  {len(categorical_features)} cat,"
      f"  {len(extra_features)} nouvelles)")


# =============================================================================
# 4. PREPROCESSEUR COMMUN
#    Identique à 02 : imputer mean + StandardScaler pour num/flags/extra,
#    OneHotEncoder pour cat.
#    StandardScaler est nécessaire avant L1/ElasticNet : ces pénalités
#    s'appliquent uniformément sur tous les coefficients, donc les features
#    doivent être à la même échelle.
# =============================================================================
numeric_and_flag = numeric_features + flag_features + extra_features

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


# =============================================================================
# 5. COMPARAISON DES TROIS MODÈLES
#    Plage d'alpha élargie : 10^-3 à 10^6 (20 valeurs log-espacées).
#    Raison : Ridge a convergé à ~144 en 02 (centre de la plage 10^-3..10^4).
#    Lasso/ElasticNet tendent à demander des alpha plus petits car la
#    pénalité L1 est plus agressive (coefficients mis à zéro rapidement).
#    On étend vers le haut pour s'assurer que Ridge n'est pas tronqué.
#
#    ElasticNet : l2_ratio=0.5 est le point médian L1/L2.
#    On pourrait aussi chercher l2_ratio mais ça multiplierait la grille --
#    on garde 0.5 comme valeur représentative.
# =============================================================================
print("\n" + "=" * 60)
print("5. GridSearchCV pour Ridge / Lasso / ElasticNet")
print("=" * 60)

alphas = np.logspace(-3, 6, 25)

MODELS = {
    "Ridge     (L2)": {
        "estimator_key": "model",
        "estimator":     Ridge(),
        "param_grid":    {"model__alpha": alphas},
    },
    "Lasso     (L1)": {
        "estimator_key": "model",
        "estimator":     Lasso(max_iter=5000),
        "param_grid":    {"model__alpha": alphas},
    },
    "ElasticNet (L1+L2)": {
        "estimator_key": "model",
        "estimator":     ElasticNet(l1_ratio=0.5, max_iter=5000),
        "param_grid":    {"model__alpha": alphas},
    },
}

cv_strategy = GroupKFold(n_splits=5)
results = {}   # nom -> dict de métriques

for name, cfg in MODELS.items():
    print(f"\n--- {name} ---")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        (cfg["estimator_key"], cfg["estimator"]),
    ])

    search = GridSearchCV(
        pipe,
        cfg["param_grid"],
        cv=cv_strategy,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train_clean, groups=groups)

    best_alpha   = list(search.best_params_.values())[0]
    best_pipe    = search.best_estimator_

    cv_res = cross_validate(
        best_pipe,
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

    rmse_val   = -cv_res["test_rmse"].mean()
    rmse_std   =  cv_res["test_rmse"].std()
    mae_val    = -cv_res["test_mae"].mean()
    r2_val     =  cv_res["test_r2"].mean()
    rmse_train = -cv_res["train_rmse"].mean()

    print(f"  alpha        : {best_alpha:.4g}")
    print(f"  RMSE val     : {rmse_val:.4f} +/- {rmse_std:.4f}  <-- metrique cle")
    print(f"  MAE  val     : {mae_val:.4f}")
    print(f"  R2   val     : {r2_val:.4f}")
    print(f"  RMSE train   : {rmse_train:.4f}")

    results[name] = {
        "best_alpha":   best_alpha,
        "best_pipe":    best_pipe,
        "rmse_val":     rmse_val,
        "rmse_std":     rmse_std,
        "mae_val":      mae_val,
        "r2_val":       r2_val,
        "rmse_train":   rmse_train,
    }


# =============================================================================
# 6. TABLEAU COMPARATIF
# =============================================================================
print("\n" + "=" * 60)
print("6. Tableau comparatif RMSE")
print("=" * 60)

header = f"{'Modele':<22}  {'alpha':>10}  {'RMSE val':>10}  {'(+/-)':>7}  {'RMSE train':>11}  {'R2 val':>7}"
print(header)
print("-" * len(header))

best_name = min(results, key=lambda k: results[k]["rmse_val"])

for name, r in results.items():
    marker = "  <-- MEILLEUR" if name == best_name else ""
    print(
        f"{name:<22}  {r['best_alpha']:>10.4g}  "
        f"{r['rmse_val']:>10.4f}  {r['rmse_std']:>7.4f}  "
        f"{r['rmse_train']:>11.4f}  {r['r2_val']:>7.4f}"
        f"{marker}"
    )

print()
print(f"Amelioration vs 02_ridge (RMSE=7.4255) :")
for name, r in results.items():
    delta = r["rmse_val"] - 7.4255
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<22}  {sign}{delta:.4f}")


# =============================================================================
# 7. FIT FINAL + PRÉDICTIONS DU MEILLEUR MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print(f"7. Fit final : {best_name.strip()}")
print("=" * 60)

best_pipeline = results[best_name]["best_pipe"]
best_pipeline.fit(X_train, y_train_clean)
y_pred = best_pipeline.predict(X_test)

print(f"Predictions : min={y_pred.min():.2f}  max={y_pred.max():.2f}"
      f"  mean={y_pred.mean():.2f}  std={y_pred.std():.2f}")


# =============================================================================
# 8. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 60)
print("8. Sauvegarde")
print("=" * 60)

y_pred_df = pd.DataFrame({
    "Index":  X_test_enh["Index"].values,
    "target": y_pred,
})
y_pred_df.to_csv("data/y_pred_best.csv", index=False)

print(f"OK data/y_pred_best.csv  ({len(y_pred_df):,} lignes)")
print(y_pred_df.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("Termine.")
print("=" * 60)
