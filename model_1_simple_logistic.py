import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer

#load train/test files (change file paths as needed)
train_df = pd.read_csv("Queries/Candidate_Contested_1.csv")
test_df  = pd.read_csv("Queries/Candidate_Contested_2.csv")

DROP_COLS = [
    "Win",
    "general_election_result",
    "general_votes_received",
    "total_general_votes_cast",
    "viability_score",
]

#preprocessing function

def prep(df: pd.DataFrame):
    # target
    df = df.copy()
    df["Win"] = (df["general_election_result"] == "Won General").astype(int)

    y = df["Win"].astype(int)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    return X, y


#split into X/y and train/test sets
X_train, y_train = prep(train_df)
X_test,  y_test  = prep(test_df)

#identify numeric and categorical columns
num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(exclude=["number"]).columns

#build pipelines for numeric and categorical features, then combine with ColumnTransformer
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), #changeable imputation strategy (mean, median, most_frequent, constant)
    ("scaler", StandardScaler())
])


categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), #Filling missing categorical values with "Unknown"
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


#combine numeric and categorical pipelines
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])


#build final pipeline with preprocessing and logistic regression model
clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])


#fit the model
clf.fit(X_train, y_train)

# predict on test
proba = clf.predict_proba(X_test)[:, 1]

pred = (proba >= 0.475).astype(int) #changeable threshold for classification 

# evaluation
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

