
import pandas as pd
i = 0
i+=1
print(pd.Timestamp.now(), i)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

i+=1
print(pd.Timestamp.now(), i)
# Load
df = pd.read_csv("Queries/New_Query_2026_01_26_12_07pm.csv")
i+=1
print(pd.Timestamp.now(), i)

# Dates
df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
i+=1
print(pd.Timestamp.now(), i)

# One row per candidate-election-office (adjust key if needed)
key_cols = ["hubspot_id", "election_date", "candidate_office"]
df = df.drop_duplicates(subset=key_cols)
i+=1 
print(pd.Timestamp.now(), i)

# Binary training set (v1)
train_df = df[df["general_election_result"].isin(["Won General", "Lost General"])].copy()
train_df["y"] = (train_df["general_election_result"] == "Won General").astype(int)
i+=1
print(pd.Timestamp.now(), i)

# Basic cleaning / feature engineering
train_df["office_level"] = train_df["office_level"].astype(str).str.upper().replace("NAN", np.nan)
train_df["office_type"]  = train_df["office_type"].astype(str).str.upper().replace("NAN", np.nan)
train_df["state"]        = train_df["state"].astype(str).str.upper().replace("NAN", np.nan)
i+=1 
print(pd.Timestamp.now(), i)

train_df["election_year"] = train_df["election_date"].dt.year
train_df["election_month"] = train_df["election_date"].dt.month

# IMPORTANT: drop obvious leakage columns for pre-election prediction
leak_cols = ["general_votes_received", "total_general_votes_cast", "general_election_result"]
X = train_df.drop(columns=leak_cols + ["y"])
y = train_df["y"]
i+=1 
print(pd.Timestamp.now(), i)

# Choose features
cat_cols = ["state", "office_level", "office_type"]
num_cols = ["viability_score", "election_year", "election_month"]
text_col = "candidate_office"  # can also add official_office_name later
i+=1 
print(pd.Timestamp.now(), i)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
        ("txt", TfidfVectorizer(min_df=5, ngram_range=(1,2)), text_col),
    ],
    remainder="drop"
)
i+=1 
print(pd.Timestamp.now(), i)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
i+=1 
print(pd.Timestamp.now(), i)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", model)
])
i+=1 
print(pd.Timestamp.now(), i)

# Time split example: train on <= 2022, test on >= 2023 (adjust)
cut_year = 2023
train_mask = X["election_year"] < cut_year
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test   = X[~train_mask], y[~train_mask]

pipe.fit(X_train, y_train)
i+=1 
print(pd.Timestamp.now(), i)

p_test = pipe.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("PR-AUC:", average_precision_score(y_test, p_test))
print("Brier:", brier_score_loss(y_test, p_test))
i+=1 
print(pd.Timestamp.now(), i)

# Score everyone (including unknown outcomes)
score_df = df.drop_duplicates(subset=key_cols).copy()
score_df["election_date"] = pd.to_datetime(score_df["election_date"], errors="coerce")
score_df["election_year"] = score_df["election_date"].dt.year
score_df["election_month"] = score_df["election_date"].dt.month

score_df["office_level"] = score_df["office_level"].astype(str).str.upper().replace("NAN", np.nan)
score_df["office_type"]  = score_df["office_type"].astype(str).str.upper().replace("NAN", np.nan)
score_df["state"]        = score_df["state"].astype(str).str.upper().replace("NAN", np.nan)
i+=1 
print(pd.Timestamp.now(), i)

# Ensure the same columns exist
for c in ["viability_score", "candidate_office"]:
    if c not in score_df.columns:
        score_df[c] = np.nan
        i+=1 
        print(pd.Timestamp.now(), i)

# Predict win probabilities
# (pipe expects the same feature columns; drop leak cols if present)
score_X = score_df.copy()
for c in leak_cols:
    if c in score_X.columns:
        score_X = score_X.drop(columns=[c])
        i+=1 
        print(pd.Timestamp.now(), i)

score_df["pred_win_prob"] = pipe.predict_proba(score_X)[:, 1]
score_df["pred_win_pct"] = (100 * score_df["pred_win_prob"]).round(1)
i+=1 
print(pd.Timestamp.now(), i)

out_cols = ["hubspot_id", "election_date", "candidate_office", "pred_win_pct", "pred_win_prob"]
scored = score_df[out_cols].sort_values("pred_win_prob", ascending=False)
print(scored.head(20))
i+=1 
print(pd.Timestamp.now(), i)
