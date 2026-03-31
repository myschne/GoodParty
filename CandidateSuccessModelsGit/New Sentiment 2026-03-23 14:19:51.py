############################################################
############ALL FUNCTIONS USING VADER SENTIMENT#############
############################################################

# %%
# 1. Install the required libraries
%pip install vaderSentiment textblob

# 2. Download the TextBlob dictionary (Corpora)
import os
os.system("python -m textblob.download_corpora")

# 3. Force the restart so the imports below actually work
dbutils.library.restartPython()

# %%
import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize Global Variables
analyzer = SentimentIntensityAnalyzer()
GROUP_COLS = ["hubspot_id", "election_date"]

# ---------------------------------------------------------
# 1. VADER THEME PROFILE
# ---------------------------------------------------------
def add_candidate_theme_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _get_scores(text):
        if pd.isna(text) or str(text).strip() == "":
            return pd.Series({'pos': 0.0, 'neg': 0.0, 'neu': 1.0})
        vs = analyzer.polarity_scores(str(text))
        return pd.Series({'pos': vs['pos'], 'neg': vs['neg'], 'neu': vs['neu']})

    scores = df["script"].apply(_get_scores)
    temp_df = pd.concat([df[GROUP_COLS], scores], axis=1)
    agg = temp_df.groupby(GROUP_COLS).mean().reset_index()
    agg.columns = [c if c in GROUP_COLS else f"pct_theme_{c}" for c in agg.columns]
    return df.merge(agg, on=GROUP_COLS, how="left")

# ---------------------------------------------------------
# 2. TEXTBLOB PROFILE (Polarity & Subjectivity)
# ---------------------------------------------------------
def add_candidate_textblob_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _get_blob_scores(text):
        if pd.isna(text) or str(text).strip() == "":
            return pd.Series({'tb_polarity': 0.0, 'tb_subjectivity': 0.0})
        blob = TextBlob(str(text))
        return pd.Series({'tb_polarity': blob.sentiment.polarity, 'tb_subjectivity': blob.sentiment.subjectivity})

    scores = df["script"].apply(_get_blob_scores)
    temp_df = pd.concat([df[['hubspot_id']], scores], axis=1)
    agg = temp_df.groupby('hubspot_id').mean().reset_index()
    return df.merge(agg, on='hubspot_id', how="left")

# ---------------------------------------------------------
# 3. AUTHENTICITY & PERSPECTIVE (The "You vs I" logic)
# ---------------------------------------------------------
def add_intro_authenticity_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    intro_pattern = r"\b(hi|hello|hey)?\s*(i am|i'm|this is)\b"
    df["_is_intro"] = df["script"].str.lower().str.contains(intro_pattern, regex=True, na=False).astype(int)
    agg = df.groupby(GROUP_COLS)["_is_intro"].mean().reset_index()
    agg.rename(columns={"_is_intro": "score_personal_intro_pct"}, inplace=True)
    return df.merge(agg, on=GROUP_COLS, how="left").drop(columns=["_is_intro"])

def add_voter_focus_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    i_pattern, you_pattern = r"\b(i|me|my|mine)\b", r"\b(you|your|yours)\b"
    df["_i_cnt"] = df["script"].str.lower().str.count(i_pattern).fillna(0)
    df["_u_cnt"] = df["script"].str.lower().str.count(you_pattern).fillna(0)
    agg = df.groupby(GROUP_COLS).agg({'_i_cnt':'sum', '_u_cnt':'sum'}).reset_index()
    agg["score_voter_focus_ratio"] = (agg["_u_cnt"] + 1) / (agg["_i_cnt"] + 1)
    return df.merge(agg[GROUP_COLS + ["score_voter_focus_ratio"]], on=GROUP_COLS, how="left")


#################################################
#########FUNCITON USING TEXT BLOB################
#################################################
# %%
# 1. Install the package
%pip install textblob

# 2. Download the 'corpora' (the logic/dictionary it needs)
import os
os.system("python -m textblob.download_corpora")

# 3. Restart to make sure the new library is visible
dbutils.library.restartPython()

# %%
from textblob import TextBlob
import pandas as pd

def add_candidate_textblob_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def _get_blob_scores(text):
        # Handle Nulls or empty strings
        if pd.isna(text) or str(text).strip() == "":
            return pd.Series({'tb_polarity': 0.0, 'tb_subjectivity': 0.0})
        
        blob = TextBlob(str(text))
        return pd.Series({
            'tb_polarity': blob.sentiment.polarity,      # -1.0 to 1.0
            'tb_subjectivity': blob.sentiment.subjectivity # 0.0 to 1.0
        })

    # Apply scores and average by hubspot_id
    scores = df["script"].apply(_get_blob_scores)
    temp_df = pd.concat([df[['hubspot_id']], scores], axis=1)
    
    agg = temp_df.groupby('hubspot_id').mean().reset_index()
    return df.merge(agg, on='hubspot_id', how="left")