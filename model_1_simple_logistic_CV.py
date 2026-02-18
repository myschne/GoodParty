import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

THRESHOLD = 0.475

FOLD_PATHS = [
    "Queries/Individual_Candidate_Query_1.csv",
    "Queries/Individual_Candidate_Query_2.csv",
    "Queries/Individual_Candidate_Query_3.csv",
    "Queries/Individual_Candidate_Query_4.csv",
    "Queries/Individual_Candidate_Query_5.csv",
]

DROP_COLS = [
    "Win",
    "general_election_result",
    "general_votes_received",
    "total_general_votes_cast",
    "viability_score_mean",
    "viability_score_max",
    "election_date",
    "latest_outreach"
]

# -----------------------
# Viability (0–5) + bucket labels (1–5 categories)
# -----------------------
VIAB_LABELS = ["No Chance", "Unlikely to Win", "Has a Chance", "Likely to Win", "Frontrunner"]

def proba_to_viability_score(proba: np.ndarray) -> np.ndarray:
    return 5.0 * proba

def viability_score_to_bucket(v: np.ndarray) -> pd.Categorical:
    return pd.cut(
        v,
        bins=[0, 1, 2, 3, 4, 5],
        labels=VIAB_LABELS,
        right=False,
        include_lowest=True
    )

# -----------------------
# Preprocessing function (feature engineering)
# -----------------------

def prep(df: pd.DataFrame):
    # target
    df = df.copy()
    #df["Win"] = (df["general_election_result"] == "Won General").astype(int)
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
    df["latest_outreach"] = pd.to_datetime(df["latest_outreach"], errors="coerce")

    # simple components
    df["election_year"]  = df["election_date"].dt.year
    df["election_month"] = df["election_date"].dt.month
    df["election_doy"]   = df["election_date"].dt.dayofyear
    df["is_midterm"] = ((df["election_year"] % 4 != 0) & (df["election_year"] % 2 == 0)).astype(int)
    df["is_presidential"] = (df["election_year"] % 4 == 0).astype(int)
    delta_days = (df["election_date"] - df["latest_outreach"]).dt.days
    df["days_between_outreach_and_election"] = delta_days.fillna(99999).astype("Int64")




    y = df["Win"].astype(int)
    X = df.drop(columns=DROP_COLS, errors="ignore")

    return X, y


# -----------------------
# Define pipelines ONCE (reused each fold)
# -----------------------
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# -----------------------
# Load folds
# -----------------------
fold_dfs = [pd.read_csv(p) for p in FOLD_PATHS]

# normalize ids
fold_sets = [
    set(df["hubspot_id"].astype(str).str.strip())
    for df in fold_dfs
]
total_overlap = 0
for i in range(len(fold_sets)):
    for j in range(i + 1, len(fold_sets)):
        inter = fold_sets[i].intersection(fold_sets[j])
        if inter:
            print(f"Overlap between fold {i} and {j}: {len(inter)} hubspot_id(s)")
            total_overlap += len(inter)

print("Total cross-fold hubspot_id overlap:", total_overlap)


# -----------------------
# Batch-as-fold CV
# -----------------------
K = len(fold_dfs)
fold_aucs = []

y_all = []
proba_all = []
rows_all = [] 
coef_series_list = []

for k in range(K):
    test_df = fold_dfs[k].copy()
    train_df = pd.concat([fold_dfs[i] for i in range(K) if i != k], ignore_index=True)

    X_train, y_train = prep(train_df)
    X_test,  y_test  = prep(test_df)

    # IMPORTANT: compute column lists from THIS fold's training data
    num_cols = X_train.select_dtypes(include=["number"]).columns
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    clf = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    clf.fit(X_train, y_train)

    # ---- collect per-fold feature importances (coef) ----
    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # numeric feature names
    num_names = list(pre.named_transformers_["num"].feature_names_in_) \
        if hasattr(pre.named_transformers_["num"], "feature_names_in_") else list(num_cols)

    # categorical feature names after one-hot
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols)

    feature_names = np.r_[num_names, cat_names]
    coefs = model.coef_.ravel()

    coef_series_list.append(pd.Series(coefs, index=feature_names, name=f"fold_{k}"))

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fold_aucs.append(auc)

    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, pred)

    print(f"\nFold {k+1}/{K} ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)

    y_all.append(y_test.to_numpy())
    proba_all.append(proba)

    # attach predictions back to the original (un-prepped) test rows
    scored = test_df.copy()
    scored["proba_win"] = proba
    scored["pred_win"] = pred
    rows_all.append(scored)

# -----------------------
# Aggregate metrics
# -----------------------
y_all = np.concatenate(y_all)
proba_all = np.concatenate(proba_all)
pred_all = (proba_all >= THRESHOLD).astype(int)

print("\n============================")
print("5-Fold Batch-CV Summary")
print("============================")
print("Fold AUCs:", [round(a, 4) for a in fold_aucs])
print("Mean AUC :", float(np.mean(fold_aucs)))
print("Std AUC  :", float(np.std(fold_aucs)))

print("\nPooled ROC-AUC:", roc_auc_score(y_all, proba_all))
print("Pooled confusion matrix:\n", confusion_matrix(y_all, pred_all))
print(classification_report(y_all, pred_all))    


# -----------------------
# Build results dataframe w/ viability (0–5) and 1–5 bucket classification
# -----------------------
results = pd.concat(rows_all, ignore_index=True)

results["viability_score_model"] = proba_to_viability_score(results["proba_win"].to_numpy())
results["viability_bucket_model"] = viability_score_to_bucket(results["viability_score_model"].to_numpy())
results["viability_bucket_num"] = results["viability_bucket_model"].cat.codes + 1  # 1..5

print("\nViability bucket distribution (model):")
print(results["viability_bucket_model"].value_counts(dropna=False))

# Example preview
print("\nSample scored rows:")
print(results[[
    "hubspot_id",
    "proba_win",
    "pred_win",
    "viability_score_model",
    "viability_bucket_num",
    "viability_bucket_model"
]].head(20).to_string(index=False))


# -----------------------
# Compare to original viability_score (0–5) from dataset
# -----------------------

# Original score -> bucket
results["viability_bucket_orig"] = viability_score_to_bucket(results["viability_score_mean"].to_numpy())
results["viability_bucket_orig_num"] = results["viability_bucket_orig"].cat.codes + 1  # 1..5

# Crosstab (orig vs model)
ct = pd.crosstab(
    results["viability_bucket_orig"],
    results["viability_bucket_model"],
    dropna=False
)
print("\nCrosstab: original viability bucket (rows) vs model bucket (cols)")
print(ct)

# Exact agreement rate
agree = (results["viability_bucket_orig"] == results["viability_bucket_model"]).mean()
print("\nExact bucket agreement:", round(float(agree), 4))

# How far off (0..4 buckets away)
results["bucket_distance"] = (
    results["viability_bucket_orig_num"] - results["viability_bucket_num"]
).abs()

print("\nBucket distance distribution (0=match, 1=adjacent, ...):")
print(results["bucket_distance"].value_counts(dropna=False).sort_index())

print("\nMean bucket distance:", round(float(results["bucket_distance"].mean()), 4))

#compare continuous scores
valid = results[["viability_score_mean", "viability_score_model"]].dropna()
corr = valid["viability_score_mean"].corr(valid["viability_score_model"])
print("\nCorrelation between original viability_score and model-derived viability_score:", round(float(corr), 4))


# ======================== Fold-averaged Feature Importance =========================
# Align features across folds (union), fill missing with 0
coef_df = pd.concat(coef_series_list, axis=1).fillna(0.0)  # rows=features, cols=folds

imp = pd.DataFrame({
    "mean_coef": coef_df.mean(axis=1),
    "std_coef": coef_df.std(axis=1),
    "mean_abs_coef": coef_df.abs().mean(axis=1),
    "sign_consistency": (np.sign(coef_df).replace(0, np.nan).mean(axis=1)).abs()
}).sort_values("mean_abs_coef", ascending=False)

# Top features overall (by average absolute effect)
print("\nTop 25 features by mean absolute coefficient across folds:")
print(imp.head(25).to_string())

# Top positive and negative on average
print("\nTop 25 features pushing toward Win=1 on average:")
print(imp.sort_values("mean_coef", ascending=False).head(25)[["mean_coef","std_coef","mean_abs_coef","sign_consistency"]].to_string())

print("\nTop 25 features pushing toward Win=0 on average:")
print(imp.sort_values("mean_coef", ascending=True).head(25)[["mean_coef","std_coef","mean_abs_coef","sign_consistency"]].to_string())

from viz_outreach_model_outputs import make_all_plots
make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp, outdir="viz_outputs")