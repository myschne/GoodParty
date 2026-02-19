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

i = 0

csv_list = [
    "Queries/Individual_Candidate_Query_1.csv",
    "Queries/Individual_Candidate_Query_2.csv",
    "Queries/Individual_Candidate_Query_3.csv",
    "Queries/Individual_Candidate_Query_4.csv",
    "Queries/Individual_Candidate_Query_5.csv",
]

DROP_COLS = [
    "general_election_result",
    "general_votes_received",
    "total_general_votes_cast",
    "viability_score_mean",
    "viability_score_max",
    "election_date",
    "latest_outreach",
    "election_year",
    "election_dow",
]

for csv in csv_list:
    # target
    i += 1
    df = pd.read_csv(csv)
    df = df.copy()
    #df["Win"] = (df["general_election_result"] == "Won General").astype(int)
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
    df["latest_outreach"] = pd.to_datetime(df["latest_outreach"], errors="coerce")

    # simple components
    df["election_year"]  = df["election_date"].dt.year
    df["election_month"] = df["election_date"].dt.month
    df["is_midterm"] = ((df["election_year"] % 4 != 0) & (df["election_year"] % 2 == 0)).astype(int)
    df["is_presidential"] = (df["election_year"] % 4 == 0).astype(int)
    delta_days = (df["election_date"] - df["latest_outreach"]).dt.days
    df["days_between_outreach_and_election"] = delta_days.fillna(99999).astype("Int64")
    df["is_normal_election"] = (
        (df["election_date"].dt.month == 11) &
        (df["election_date"].dt.dayofweek == 1) &   # Monday=0, Tuesday=1u, ...
        (df["election_date"].dt.day.between(2, 8))
    ).astype(int)
    #Make dow categorical
    names = ["mon","tue","wed","thu","fri","sat","sun"]
    doy_dummies = pd.get_dummies(
        df["election_date"].dt.day_name().str[:3].str.lower(),
        prefix="election_dow",
        dtype=int
    ).reindex(columns=[f"election_dow_{n}" for n in names], fill_value=0)

    df = pd.concat([df, doy_dummies], axis=1)

    df = df.drop(columns=DROP_COLS, errors="ignore")

    df.to_csv(f"prepped_candidates_{i}.csv", index=False)




    