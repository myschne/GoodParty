"""
Feature engineering utilities for the Candidate Success modeling pipeline.

This module contains the transformation logic used to convert raw
candidate-election data into a cleaner, model-ready feature set. It includes
helpers for viability score mapping, office-level normalization, state and
region standardization, engineered election/outreach features, and final
construction of the modeling matrix.

Main responsibilities of this module:
- map predicted probabilities into project viability scores and labels
- standardize noisy office-level text into a small set of categories
- standardize state values into USPS codes and broader regions
- create election timing and outreach recency features
- create race-structure and incumbency features
- build the final feature matrix X and target y for training or scoring

This file is central to consistency across the pipeline because the same
feature-building logic is used during both training and inference.
"""


import numpy as np
import pandas as pd
import re
from config import ALPHA, DROP_COLS, VIAB_LABELS

# -----------------------
# Viability (0–5) + bucket labels (1–5 categories)
# -----------------------

def proba_to_viability_score(proba: np.ndarray) -> np.ndarray:
    """Scale probability to a 0-5 viability score."""
    return 5.0 * proba

def viability_score_to_bucket(v: np.ndarray) -> pd.Categorical:
    """Convert a continuous viability score to 1–5 categorical buckets."""
    return pd.cut(
        v,
        bins=[0, 1, 2, 3, 4, 5],
        labels=VIAB_LABELS,
        right=False,
        include_lowest=True
    )

# -----------------------
# Preprocessing functions (feature engineering): normalize and categorize office level
# -----------------------
def norm_office_level(x):
    """Standardize office_level strings: lowercase, strip spaces, remove punctuation."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)                 # collapse whitespace (handles "CITY ")
    s = re.sub(r"[^\w\s-]", "", s)             # drop punctuation, keep hyphen
    return s

# mapping to buckets
OFFICE_LEVEL_MAP = {
    # null-ish / junk
    "": np.nan,
    "null": np.nan,
    "n/a": np.nan,
    "na": np.nan,
    "undefined": np.nan,
    "2": np.nan,
    "3": np.nan,

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
    """Clean office_level column: normalize, map to categories, fallback rules."""
    s = series.map(norm_office_level)

    # direct map first
    out = s.map(OFFICE_LEVEL_MAP)

    # fallback rules for anything not caught by exact mapping
    # (handles unexpected variants like "Federal Senate", "City Council", etc.)
    still = out.isna() & s.notna() & (s != "")
    out.loc[still & s.str.contains(r"\bfederal\b|\bpresident\b", regex=True)] = "federal"
    out.loc[still & s.str.contains(r"\bstate\b|\bstatewide\b|\blegisl", regex=True)] = "state"
    out.loc[still & s.str.contains(r"\blocal\b|\bcity\b|\bmunicipal\b|\bcounty\b|\btown\b|\btownship\b|\bregional\b", regex=True)] = "local"

    # anything remaining -> other
    out = out.fillna("other")
    
    return out.astype("category")


# -----------------------
# State cleaning and region mapping
# -----------------------

US_STATE_CODES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
    'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
    'VT','VA','WA','WV','WI','WY','DC'
}

US_TERRITORY_CODES = {'AS','GU','MP','PR','VI','PW'}
US_CODES_ALL = US_STATE_CODES | US_TERRITORY_CODES

STATE_NAME_TO_USPS = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT",
    "DELAWARE":"DE","DISTRICT OF COLUMBIA":"DC","FLORIDA":"FL","GEORGIA":"GA","HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL",
    "INDIANA":"IN","IOWA":"IA","KANSAS":"KS","KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD",
    "MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS","MISSOURI":"MO","MONTANA":"MT",
    "NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ","NEW MEXICO":"NM","NEW YORK":"NY",
    "NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK","OREGON":"OR","PENNSYLVANIA":"PA",
    "RHODE ISLAND":"RI","SOUTH CAROLINA":"SC","SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT",
    "VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA","WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY",

    # Territories (if present as names)
    "AMERICAN SAMOA":"AS","GUAM":"GU","NORTHERN MARIANA ISLANDS":"MP","PUERTO RICO":"PR",
    "VIRGIN ISLANDS":"VI","U.S. VIRGIN ISLANDS":"VI","PALAU":"PW",
}

# Common variants/typos -> either USPS or canonical name handled above
STATE_ALIASES = {
    "DELEWARE": "DE",
    "WASHINGTON DC": "DC",
    "WASHINGTON D.C.": "DC",
    "DISTRICT OF COLUMBIA": "DC",
    # add more if discovered
}

REGION_MAP = {
    "NORTHEAST": {"CT","ME","MA","NH","RI","VT","NJ","NY","PA"},
    "MIDWEST":   {"IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD"},
    "SOUTH":     {"DE","FL","GA","MD","NC","SC","VA","WV","AL","KY","MS","TN","AR","LA","OK","TX","DC"},
    "WEST":      {"AZ","CO","ID","MT","NV","NM","UT","WY","AK","CA","HI","OR","WA"},
}
def norm_state(x) -> str:
    """Normalize state string: uppercase, remove periods, collapse whitespace."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    s = re.sub(r"\s+", " ", s)          # collapse whitespace
    s = s.replace(".", "")              # remove periods (D.C. -> DC)
    return s.upper()

def clean_state_to_usps(series: pd.Series, keep_only_us_territories: bool = True) -> pd.Series:
    """
    Returns USPS code for US states/DC/territories.
    If keep_only_us_territories=True, anything else becomes "Unknown"
    """
    s = series.map(norm_state)

    def _map_one(val: str):
        if val == "" or val in {"NULL", "N/A", "NA", "UNDEFINED"}:
            return "Unknown"

        # direct alias (typos/variants)
        if val in STATE_ALIASES:
            code = STATE_ALIASES[val]
            return code if code in US_CODES_ALL else "Unknown"

        # if it's already a code, only accept if it's a real US code
        if len(val) == 2 and val.isalpha():
            return val if val in US_CODES_ALL else ("Unknown" if keep_only_us_territories else val)

        # full name to code
        if val in STATE_NAME_TO_USPS:
            return STATE_NAME_TO_USPS[val]

        # otherwise: non-US or unmapped
        return "Unknown" if keep_only_us_territories else val

    out = s.map(_map_one).astype("string")
    return out.astype("category")

def state_usps_to_region(state_usps: pd.Series) -> pd.Series:
    """
    Maps USPS -> {NORTHEAST, MIDWEST, SOUTH, WEST, TERRITORY, UNKNOWN}.
    """
    def _region(code):
        if pd.isna(code) or str(code).strip() == "" or str(code).strip() == "Unknown":
            return "UNKNOWN"
        c = str(code)
        if c in US_TERRITORY_CODES:
            return "TERRITORY"
        for region, codes in REGION_MAP.items():
            if c in codes:
                return region
        return "UNKNOWN"

    return state_usps.map(_region).astype("category")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered modeling features from the raw candidate-level dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw or semi-processed candidate-election
        features. Expected columns include election timing fields, outreach
        timing fields, office/state information, and race structure variables.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with additional engineered features added.

    Feature groups created
    ----------------------
    Election timing features
        - election_year
        - election_month
        - is_midterm
        - is_presidential
        - is_normal_election
        - election_dow_* dummy columns

    Outreach recency features
        - days_between_outreach_and_election
        - recency_weighted_days
        - recency_election_interaction

    Geographic / office standardization
        - office_level_clean
        - state_usps
        - region

    Race structure features
        - number_avail_seats
        - number_of_opponents_num
        - competitiveness

    Incumbency features
        - incumbency_status

    Notes
    -----
    - The function works on a copy of the input dataframe.
    - election_date and latest_outreach are coerced to datetime.
    - Missing outreach timing is assigned a large placeholder value
      (99999 days) before recency weighting.
    - Blank string values in text columns are replaced with 'Unknown'.
    """
    df = df.copy()

    # Convert date fields to pandas datetime; invalid values become NaT.
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
    df["latest_outreach"] = pd.to_datetime(df["latest_outreach"], errors="coerce")

    # -----------------------------------------------------
    # Election calendar features
    # -----------------------------------------------------
    # Extract year and month from the election date.
    df["election_year"] = df["election_date"].dt.year
    df["election_month"] = df["election_date"].dt.month

    # Flag midterm election years:
    # even-numbered years that are not presidential election years.
    df["is_midterm"] = ((df["election_year"] % 4 != 0) & (df["election_year"] % 2 == 0)).astype(int)

    # Flag presidential election years:
    # years divisible by 4.
    df["is_presidential"] = (df["election_year"] % 4 == 0).astype(int)

    # -----------------------------------------------------
    # Outreach recency features
    # -----------------------------------------------------
    # Measure time gap between last outreach and election day.
    delta_days = (df["election_date"] - df["latest_outreach"]).dt.days

    # Fill missing differences with a very large value so that
    # missing / very old outreach gets very low recency weight.
    df["days_between_outreach_and_election"] = delta_days.fillna(99999).astype("Int64")

    # Exponential decay weighting:
    # smaller day gaps -> larger weights, larger gaps -> smaller weights.
    df["recency_weighted_days"] = np.exp(-ALPHA * df["days_between_outreach_and_election"])

    # Interaction between outreach volume and recency:
    # emphasizes candidates with many recent outreach rows.
    df["recency_election_interaction"] = df["n_outreach_rows"] * df["recency_weighted_days"]

    # -----------------------------------------------------
    # Election date pattern feature
    # -----------------------------------------------------
    # Flag “normal” U.S. election timing:
    # Tuesday in November, with day of month between 2 and 8.
    df["is_normal_election"] = (
        (df["election_date"].dt.month == 11) &
        (df["election_date"].dt.dayofweek == 1) &
        (df["election_date"].dt.day.between(2, 8))
    ).astype(int)

    # -----------------------------------------------------
    # Day-of-week dummy features
    # -----------------------------------------------------
    # Build one-hot encoded indicators for election day-of-week.
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    dow = pd.get_dummies(
        df["election_date"].dt.day_name().str[:3].str.lower(),
        prefix="election_dow",
        dtype=int
    ).reindex(columns=[f"election_dow_{n}" for n in names], fill_value=0)
    df = pd.concat([df, dow], axis=1)

    # -----------------------------------------------------
    # Standardize office level and state
    # -----------------------------------------------------
    # Clean office level into a more consistent categorical form.
    df["office_level_clean"] = clean_office_level(df["office_level"])

    # Convert state names / values into USPS two-letter codes.
    df["state_usps"] = clean_state_to_usps(df["state"], keep_only_us_territories=True).astype("object")

    # Map USPS code to broader Census-style region.
    df["region"] = state_usps_to_region(df["state_usps"]).astype("object")

    # -----------------------------------------------------
    # Race structure / competitiveness features
    # -----------------------------------------------------
    # Clean seat count into numeric form.
    df["number_avail_seats"] = pd.to_numeric(
        df["seats_available"].replace({"null": np.nan, 0E-10: 0}),
        errors="coerce"
    )

    # Clean opponent count into numeric form.
    # "10+" is capped at 10 for modeling convenience.
    df["number_of_opponents_num"] = pd.to_numeric(
        df["number_of_opponents"].replace({"10+": 10, "null": np.nan}),
        errors="coerce"
    )

    # Competitiveness ratio:
    # total candidates competing per available seat.
    # Adding 1 includes the focal candidate in the race size.
    df["competitiveness"] = np.where(
        df["number_of_opponents_num"].notna() &
        df["number_avail_seats"].notna() &
        (df["number_avail_seats"] != 0),
        (df["number_of_opponents_num"] + 1) / df["number_avail_seats"],
        np.nan
    )

    # -----------------------------------------------------
    # Text cleanup
    # -----------------------------------------------------
    # Replace blank strings in text columns with "Unknown"
    # so they are retained as an explicit category.
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].replace(r"^\s*$", "Unknown", regex=True)

    # -----------------------------------------------------
    # Incumbency status feature
    # -----------------------------------------------------
    # Collapse incumbent / open seat binary flags into a single
    # mutually exclusive categorical feature.
    df["incumbency_status"] = np.select(
        [
            df["incumbent"] == 1,
            (df["incumbent"] == 0) & (df["open_seat"] == 0),
            (df["incumbent"] == 0) & (df["open_seat"] == 1),
        ],
        ["is incumbent", "is challenger", "open seat"],
        default="Unknown"
    )
    # -----------------------------------------------------
    # Squared features
    # -----------------------------------------------------

    squared_features = [
        "days_between_outreach_and_election",
        "n_outreach_rows",
        "number_of_opponents_num",
        "competitiveness",
    ]

    for col in squared_features:
        if col in df.columns:
            df[f"{col}_sq"] = pd.to_numeric(df[col], errors="coerce") ** 2

    return df

def aggregate_message_level_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    df = df.copy()

    df["hubspot_id"] = df["hubspot_id"].fillna("Unknown").astype(str).str.strip()
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
    df["outreach_date"] = pd.to_datetime(df["outreach_date"], errors="coerce")

    df["valid_outreach_date"] = df["outreach_date"].where(
        df["outreach_date"] <= df["election_date"]
    )

    group_cols = ["hubspot_id", "election_date"]

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            state=("state", "first"),
            office_level=("office_level", "first"),
            office_type=("office_type", "first"),
            open_seat=("open_seat", "first"),
            incumbent=("incumbent", "first"),
            number_of_opponents=("number_of_opponents", "first"),
            partisan_type=("is_partisan", "first"),
            seats_available=("seats_available", "max"),
            viability_score_mean=("viability_score", "mean"),
            viability_score_max=("valid_outreach_date", "max"),
            latest_outreach=("outreach_date", "max"),
            n_outreach_rows=("hubspot_id", "size"),
            score_theme_trust_pct=("score_theme_trust_pct", "first"),
            score_theme_hope_pct=("score_theme_hope_pct", "first"),
            score_theme_fear_pct=("score_theme_fear_pct", "first"),
            score_theme_anger_pct=("score_theme_anger_pct", "first"),
            score_candidate_authenticity=("score_candidate_authenticity", "first"),
            score_perspective_candidate_avg=("score_perspective_candidate_avg", "first"),
            score_perspective_voter_avg=("score_perspective_voter_avg", "first"),
            **({"Win": ("Win", "max")} if training else {})
        )
        .reset_index()
    )

    type_counts = (
        df.loc[df["outreach_type"].notna() & (df["outreach_type"].astype(str).str.strip() != "")]
        .groupby(group_cols + ["outreach_type"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    most_common_type = (
        type_counts.sort_values(
            ["hubspot_id", "election_date", "n", "outreach_type"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(group_cols)
        .rename(columns={"outreach_type": "most_common_outreach_type"})
        [group_cols + ["most_common_outreach_type"]]
    )

    out = out.merge(most_common_type, on=group_cols, how="left")

    return out


def split_X_y(df: pd.DataFrame, target_col: str = "Win"):
    """
    Build engineered features, separate target from predictors,
    and return both the modeling matrix and the fully featured dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe at the candidate-election level.
    target_col : str, default="Win"
        Name of the target column to extract. If the target column is not
        present, y is returned as None.

    Returns
    -------
    tuple
        (X, y, df_featured)

        X : pd.DataFrame
            Predictor matrix after feature engineering and dropping columns
            listed in DROP_COLS.
        y : pd.Series or None
            Target vector if target_col exists in the dataframe,
            otherwise None.
        df_featured : pd.DataFrame
            Fully feature-engineered dataframe before DROP_COLS filtering.

    Notes
    -----
    - build_features() is applied first.
    - DROP_COLS is used to remove identifiers, leakage variables,
      and raw fields replaced by engineered features.
    - target_col is removed from X even if it is not already in DROP_COLS.
    """
    # Apply all feature engineering steps first.
    df = build_features(df)

    # Extract target if available; return None for scoring datasets.
    y = df[target_col].astype(int) if target_col in df.columns else None

    # Remove columns excluded from modeling.
    X = df.drop(columns=DROP_COLS, errors="ignore")

    # Extra safety: ensure target is not present in feature matrix.
    if target_col in X.columns:
        X = X.drop(columns=[target_col], errors="ignore")

    return X, y, df