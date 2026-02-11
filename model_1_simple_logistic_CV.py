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
    "viability_score",
    "election_date"
]

def prep(df: pd.DataFrame):
    # target
    df = df.copy()
    #df["Win"] = (df["general_election_result"] == "Won General").astype(int)
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")

    # simple components
    df["election_year"]  = df["election_date"].dt.year
    df["election_month"] = df["election_date"].dt.month
    df["election_doy"]   = df["election_date"].dt.dayofyear

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

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fold_aucs.append(auc)

    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, pred)

    print(f"\nFold {k+1}/{K} ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)

    y_all.append(y_test.to_numpy())
    proba_all.append(proba)

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
