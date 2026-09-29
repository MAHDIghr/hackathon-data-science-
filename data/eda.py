# %% [markdown]
# # EDA: Parkinson's Disease Motor Score Dataset
#
# Exploratory data analysis of the IBM/Probabl hackathon dataset, run
# before designing any model.
#
# - **Raw data** (`data/*.csv`) is read-only — this file never modifies it.
# - **Outputs** go under `data/`: `eda_<table>.html` per table, and
#   `eda.md` prose narrative with modelling implications.
#
# Execute via the shared runner:
#   pixi run -e agent python \
#     .agents/skills/audit-ml-pipeline/scripts/run_cells.py data/eda.py

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

EDA_DIR = Path(__file__).parent
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Four CSV files from Kaggle: feature tables for train and test,
# the training target, and the submission template.

# %%
X_train = pd.read_csv(EDA_DIR / "X_train.csv")
y_train = pd.read_csv(EDA_DIR / "y_train.csv")
X_test = pd.read_csv(EDA_DIR / "X_test.csv")

# Merge features and target for a single annotated train table.
train = X_train.merge(y_train, on="Index")
{"X_train": X_train.shape, "y_train": y_train.shape, "X_test": X_test.shape, "train_merged": train.shape}

# %% [markdown]
# ## Table overview — train (features + target)
#
# Full per-column report: types, distributions, missingness, cardinality.
# The HTML file is the interactive artefact for the team.

# %%
report_train = skrub.TableReport(train, title="Parkinson's train", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report_train.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": round(col.get("null_proportion") or 0, 3),
        "n_unique": col.get("nunique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}

# %% [markdown]
# ## Table overview — test
#
# Verify that missingness patterns and column set match train.

# %%
report_test = skrub.TableReport(X_test, title="Parkinson's test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
[
    {
        "column": col.get("name"),
        "null_pct": round(col.get("null_proportion") or 0, 3),
    }
    for col in summary_test.get("columns", [])
]

# %% [markdown]
# ## Target
#
# The target is `target` (debiased true OFF MDS-UPDRS motor score, 0–132).
# Regression task. Distribution shape determines whether a transform helps.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)

# %% [markdown]
# ## Structure signals
#
# Look for datetime columns (→ TimeSeriesSplit) and high-cardinality
# group-like columns (→ GroupKFold to prevent patient leakage).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (
                round((col.get("nunique") or 0) / n_rows, 4) if n_rows else None
            ),
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:8]}

# %% [markdown]
# ## Associations
#
# Strongest pairwise associations to identify candidate predictors and
# potential leakage (`off` and `on` are the biased measured scores —
# very strong predictors but NOT leaks since the task is to debias them).

# %%
skrub.column_associations(train).head(20)

# %% [markdown]
# ## Summary
#
# Findings and modelling implications are written up in `data/eda.md`.
