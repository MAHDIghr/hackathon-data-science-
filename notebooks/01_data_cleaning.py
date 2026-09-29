"""
=============================================================================
 01_data_cleaning.py  —  Nettoyage & Feature Engineering des données
=============================================================================

RÉSUMÉ DES DÉCISIONS (basé sur l'audit des données) :
------------------------------------------------------
COLONNE                 % NaN   STRATÉGIE
----------------------  ------  -------------------------------------------
gene                    32.4%   fillna("Unknown")  → catégorie propre
age_at_diagnosis         5.2%   imputer par patient via  age - median_tsd
                                (median time_since_diagnosis par cohort)
ledd                    36.6%   NaN = signal (pas encore sous médicament
                                ou info manquante) → garder NaN pour HGBR ;
                                créer flag ledd_missing
time_since_intake_on    46.4%   NaN = visite OFF-only → garder + flag
time_since_intake_off   78.8%   NaN = visite ON-only  → garder + flag
on                      29.7%   corr=0.688 avec target → garder NaN pour
                                HGBR ; créer flag on_missing
off                     42.4%   corr=0.886 avec target (STRONGEST SIGNAL)
                                → JAMAIS imputer ; créer flag off_missing
                                NaN off = visite ON-only (cliniquement)

FEATURES CRÉÉES :
  time_since_diagnosis  = age - age_at_diagnosis          (progression clé)
  prev_on               = on de la visite précédente      (lag)
  prev_off              = off de la visite précédente     (lag)
  prev_target_proxy     = off de la visite précédente     (lag du signal fort)
  delta_ledd            = diff() de ledd par patient      (évolution du traitement)
  n_visit               = numéro de visite (rang temporel)
  ledd_missing          = 1 si ledd est NaN               (flag)
  on_missing            = 1 si on est NaN                 (flag)
  off_missing           = 1 si off est NaN (visite ON-only)
  both_scores           = 1 si on ET off sont présents
=============================================================================
"""

import pandas as pd
import numpy as np

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("=" * 60)
print("1. Chargement des données")
print("=" * 60)

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

# Fusion train
visits = X_train.merge(y_train, on="Index")

print(f"Train : {X_train.shape[0]:,} lignes × {X_train.shape[1]} colonnes")
print(f"Test  : {X_test.shape[0]:,} lignes × {X_test.shape[1]} colonnes")
print(f"Target: min={visits['target'].min():.1f}  max={visits['target'].max():.1f}"
      f"  mean={visits['target'].mean():.2f}  std={visits['target'].std():.2f}")
print(f"Patients train : {visits['patient_id'].nunique():,}")
print(f"Patients test  : {X_test['patient_id'].nunique():,}")

# Vérification : aucun patient commun train/test
overlap = set(X_train["patient_id"]) & set(X_test["patient_id"])
assert len(overlap) == 0, f"LEAKAGE détecté : {overlap}"
print("OK Aucun patient commun train/test")


# =============================================================================
# 2. FONCTION DE NETTOYAGE (appliquée à train et test)
# =============================================================================

def clean_and_engineer(df: pd.DataFrame, median_tsd_by_cohort: dict, global_median_tsd: float) -> pd.DataFrame:
    """
    Nettoyage complet + feature engineering.
    Appliqué identiquement sur train et test.

    Parameters
    ----------
    df : DataFrame (train ou test, sans la colonne target)
    median_tsd_by_cohort : dict {cohort -> median time_since_diagnosis}
                           calculé UNIQUEMENT sur le train
    global_median_tsd    : float, fallback global si cohort inconnue
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 2a. Arrondir age (supprime les artefacts float 61.400000000000006)
    # -------------------------------------------------------------------------
    df["age"] = df["age"].round(2)

    # -------------------------------------------------------------------------
    # 2b. gene : NaN → "Unknown", puis catégorie
    # -------------------------------------------------------------------------
    df["gene"] = df["gene"].fillna("Unknown").astype("category")

    # -------------------------------------------------------------------------
    # 2c. cohort : catégorie propre
    # -------------------------------------------------------------------------
    df["cohort"] = df["cohort"].astype("category")

    # -------------------------------------------------------------------------
    # 2d. age_at_diagnosis : imputation par patient
    #     Principe : age_at_diagnosis est CONSTANT par patient (les 286 patients
    #     qui l'ont manquant l'ont manquant sur TOUTES leurs visites).
    #     On utilise : age_at_diagnosis = age - median_tsd(cohort)
    #     → estimation cliniquement raisonnable (durée médiane de maladie)
    # -------------------------------------------------------------------------
    missing_aad_mask = df["age_at_diagnosis"].isnull()
    if missing_aad_mask.any():
        def impute_aad(row):
            med_tsd = median_tsd_by_cohort.get(row["cohort"], global_median_tsd)
            return row["age"] - med_tsd
        df.loc[missing_aad_mask, "age_at_diagnosis"] = df[missing_aad_mask].apply(impute_aad, axis=1)

    # -------------------------------------------------------------------------
    # 2e. Feature clé : time_since_diagnosis (progression temporelle)
    # -------------------------------------------------------------------------
    df["time_since_diagnosis"] = (df["age"] - df["age_at_diagnosis"]).round(2)

    # -------------------------------------------------------------------------
    # 2f. Flags de missingness (le missingness est un signal clinique)
    # -------------------------------------------------------------------------
    df["ledd_missing"]  = df["ledd"].isnull().astype(int)
    df["on_missing"]    = df["on"].isnull().astype(int)
    df["off_missing"]   = df["off"].isnull().astype(int)    # visite ON-only si 1
    df["both_scores"]   = ((df["on"].notna()) & (df["off"].notna())).astype(int)

    # -------------------------------------------------------------------------
    # 2g. Lag features par patient (triées par âge = temps)
    #     IMPORTANT : trier avant shift pour que le lag soit temporellement correct
    # -------------------------------------------------------------------------
    df = df.sort_values(["patient_id", "age"]).reset_index(drop=True)

    df["prev_on"]    = df.groupby("patient_id")["on"].shift(1)
    df["prev_off"]   = df.groupby("patient_id")["off"].shift(1)
    df["delta_ledd"] = df.groupby("patient_id")["ledd"].diff()

    # Rang de visite par patient (1 = première visite)
    df["n_visit"] = df.groupby("patient_id").cumcount() + 1

    # -------------------------------------------------------------------------
    # 2h. Réordonner selon l'Index original (important pour l'alignement avec y)
    # -------------------------------------------------------------------------
    df = df.sort_values("Index").reset_index(drop=True)

    return df


# =============================================================================
# 3. CALCUL DES STATISTIQUES D'IMPUTATION (sur train UNIQUEMENT)
# =============================================================================
print("\n" + "=" * 60)
print("2. Calcul des statistiques d'imputation (train only)")
print("=" * 60)

has_aad = visits[visits["age_at_diagnosis"].notna()].copy()
has_aad["tsd"] = has_aad["age"] - has_aad["age_at_diagnosis"]
median_tsd_by_cohort = has_aad.groupby("cohort")["tsd"].median().to_dict()
global_median_tsd    = has_aad["tsd"].median()

print(f"Median time_since_diagnosis overall : {global_median_tsd:.2f} years")
print(f"Median time_since_diagnosis by cohort: {median_tsd_by_cohort}")


# =============================================================================
# 4. APPLICATION DU NETTOYAGE
# =============================================================================
print("\n" + "=" * 60)
print("3. Application du nettoyage")
print("=" * 60)

# Train (on garde la colonne target séparément pour ne pas l'ingérer)
X_train_clean = clean_and_engineer(
    visits.drop(columns=["target"]),
    median_tsd_by_cohort,
    global_median_tsd
)
y_train_clean = visits.set_index("Index")["target"].reindex(X_train_clean["Index"]).values

# Test
X_test_clean = clean_and_engineer(
    X_test,
    median_tsd_by_cohort,
    global_median_tsd
)

print(f"Train nettoyé : {X_train_clean.shape[0]:,} lignes × {X_train_clean.shape[1]} colonnes")
print(f"Test nettoyé  : {X_test_clean.shape[0]:,} lignes × {X_test_clean.shape[1]} colonnes")


# =============================================================================
# 5. VÉRIFICATION POST-NETTOYAGE
# =============================================================================
print("\n" + "=" * 60)
print("4. Vérification post-nettoyage")
print("=" * 60)

print("\nMissingness restante (colonnes qu'on laisse exprès pour HGBR) :")
remaining_nan = X_train_clean.isnull().mean().mul(100).round(2)
remaining_nan = remaining_nan[remaining_nan > 0]
print(remaining_nan.to_string())

print("\nNouvelles features créées :")
new_feats = ["time_since_diagnosis", "ledd_missing", "on_missing",
             "off_missing", "both_scores", "prev_on", "prev_off",
             "delta_ledd", "n_visit"]
for f in new_feats:
    n_nan = X_train_clean[f].isnull().sum()
    print(f"  {f:30s}  NaN={n_nan:,}")

print("\nage_at_diagnosis après imputation :")
print(f"  NaN restants : {X_train_clean['age_at_diagnosis'].isnull().sum()}")

print("\ngene après nettoyage :")
print(X_train_clean["gene"].value_counts().to_string())

# Vérification alignement y
assert len(y_train_clean) == len(X_train_clean), "Alignement train rompu !"
print(f"\nOK Alignement X_train / y_train : {len(X_train_clean):,} lignes")


# =============================================================================
# 6. COLONNES FEATURES RECOMMANDÉES
# =============================================================================
print("\n" + "=" * 60)
print("5. Colonnes features recommandées")
print("=" * 60)

# Colonnes à EXCLURE des features (identifiants / target)
EXCLUDE = ["Index", "patient_id", "target"]

# Features numériques (NaN gardés → HGBR les gère nativement)
NUMERIC_FEATURES = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "time_since_diagnosis",    # cle de progression
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
    "prev_on",                 # lag visite précédente
    "prev_off",                # lag visite précédente (fort signal)
    "delta_ledd",              # évolution du traitement
    "n_visit",                 # rang temporel
]

# Features catégorielles (encodées)
CATEGORICAL_FEATURES = [
    "gene",      # 5 catégories après fillna
    "cohort",    # A / B
]

# Flags binaires (missingness comme signal)
FLAG_FEATURES = [
    "ledd_missing",
    "on_missing",
    "off_missing",
    "both_scores",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES

print(f"Total features : {len(ALL_FEATURES)}")
print(f"  Numériques   : {len(NUMERIC_FEATURES)}")
print(f"  Catégorielles: {len(CATEGORICAL_FEATURES)}")
print(f"  Flags        : {len(FLAG_FEATURES)}")
print("\nListe complète :")
for f in ALL_FEATURES:
    print(f"  - {f}")


# =============================================================================
# 7. SAUVEGARDE DES DATASETS NETTOYÉS
# =============================================================================
print("\n" + "=" * 60)
print("6. Sauvegarde")
print("=" * 60)

X_train_clean.to_csv("data/X_train_clean.csv", index=False)
X_test_clean.to_csv("data/X_test_clean.csv", index=False)
pd.Series(y_train_clean, name="target").to_csv("data/y_train_clean.csv", index=False)

print("OK data/X_train_clean.csv")
print("OK data/X_test_clean.csv")
print("OK data/y_train_clean.csv")

# Export de la liste de features pour les autres notebooks
import json
feature_config = {
    "NUMERIC_FEATURES":      NUMERIC_FEATURES,
    "CATEGORICAL_FEATURES":  CATEGORICAL_FEATURES,
    "FLAG_FEATURES":         FLAG_FEATURES,
    "ALL_FEATURES":          ALL_FEATURES,
    "EXCLUDE":               EXCLUDE,
    "imputation_stats": {
        "median_tsd_by_cohort": median_tsd_by_cohort,
        "global_median_tsd":    float(global_median_tsd)
    }
}
with open("data/feature_config.json", "w") as f:
    json.dump(feature_config, f, indent=2)
print("OK data/feature_config.json  (liste de features partagee)")

print("\n" + "=" * 60)
print("Nettoyage terminé. Utilisez X_train_clean.csv pour la modélisation.")
print("=" * 60)
