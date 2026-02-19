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
import re

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
    "hubspot_id",
    "office_level"
]

def norm_office_level(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)                 # collapse whitespace (handles "CITY ")
    s = re.sub(r"[^\w\s-]", "", s)             # drop punctuation, keep hyphen
    return s

# 2) mapping to canonical buckets
OFFICE_LEVEL_MAP = {
    # null-ish / junk
    "": pd.NA,
    "null": pd.NA,
    "n/a": pd.NA,
    "na": pd.NA,
    "undefined": pd.NA,
    "2": pd.NA,
    "3": pd.NA,

    # local / municipal / city / town / township / county
    "local": "local",
    "city": "local",
    "municipal": "local",
    "town": "local",
    "township": "local",
    "county": "local",
    "regional": "local",   # if you want "regional" separate, change this

    # state
    "state": "state",
    "state1": "state",
    "statewide": "state",
    "state legislative": "state",
    "legislative": "state",  # ambiguous; if you have federal legislative too, handle separately

    # federal
    "federal": "federal",
    "federal-download": "federal",
    "presidential": "federal",  # or "presidential" if you want its own class
}

def clean_office_level(series: pd.Series) -> pd.Series:
    s = series.map(norm_office_level)

    # direct map first
    out = s.map(OFFICE_LEVEL_MAP)

    # 3) fallback rules for anything not caught by exact mapping
    # (handles unexpected variants like "Federal Senate", "City Council", etc.)
    still = out.isna() & s.notna() & (s != "")
    out.loc[still & s.str.contains(r"\bfederal\b|\bpresident\b", regex=True)] = "federal"
    out.loc[still & s.str.contains(r"\bstate\b|\bstatewide\b|\blegisl", regex=True)] = "state"
    out.loc[still & s.str.contains(r"\blocal\b|\bcity\b|\bmunicipal\b|\bcounty\b|\btown\b|\btownship\b|\bregional\b", regex=True)] = "local"

    # anything remaining -> other (or pd.NA)
    out = out.fillna("other")

    return out.astype("category")

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

    df["office_level_clean"] = clean_office_level(df["office_level"])

    df = df.drop(columns=DROP_COLS, errors="ignore")

    df.to_csv(f"prepped_candidates_{i}.csv", index=False)




    