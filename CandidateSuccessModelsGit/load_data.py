"""
Data loading utilities for training and scoring datasets.

This module loads the base outreach data from Spark SQL and enriches it with
district-level demographic features derived from the deid_voters table.

Why this logic lives here
-------------------------
The outreach table stores district information in normalized form:
    - l2_district_type
    - l2_district_name

The voter table stores district assignments in wide form:
    - one column per district type

Because the matching district columns depend on the actual district types that
appear in candidates_outreach, the cleanest approach is to build the district
feature query dynamically in Python at load time.

District features added
-----------------------
For each outreach row, this loader attempts to attach:
    - district_n_voters
    - avg_age_in_district
    - pct_female_in_district
    - pct_male_in_district
    - pct_hs_or_less_in_district
    - pct_college_grad_in_district
    - pct_postgrad_in_district

These features are joined using:
    candidates_outreach.l2_district_type == deid_voters.<matching district column name>
    candidates_outreach.l2_district_name == district value in that column

Notes
-----
- Only district columns that appear in BOTH candidates_outreach.l2_district_type
  and the deid_voters schema are used.
- This keeps the logic generic and avoids hard-coding every district type.
- If no matching district columns are found, the loader falls back to the base
  outreach data and fills district features with missing values.
"""

from __future__ import annotations
from sql_query import TRAINING_MESSAGE_QUERY, SCORING_MESSAGE_QUERY
from config import OUTREACH_TABLE, VOTER_TABLE
import pandas as pd
import numpy as np


#======================================================================
# Helpers: schema inspection
#======================================================================

def _get_outreach_district_types(spark) -> list[str]:
    """
    Return the distinct l2_district_type values that appear in outreach data.

    Output values are lowercased and stripped so they can be matched against
    deid_voters column names.
    """
    rows = spark.sql(f"""
        SELECT DISTINCT LOWER(TRIM(l2_district_type)) AS district_type
        FROM {OUTREACH_TABLE}
        WHERE l2_district_type IS NOT NULL
          AND TRIM(l2_district_type) <> ''
    """).collect()

    return sorted(
        {
            r["district_type"]
            for r in rows
            if r["district_type"] is not None and str(r["district_type"]).strip() != ""
        }
    )


def _get_voter_columns(spark) -> set[str]:
    """
    Return the deid_voters column names as a lowercase set.
    """
    return {c.lower() for c in spark.table(VOTER_TABLE).columns}


def _get_matching_district_columns(spark) -> list[str]:
    """
    Return district types that are present both:
    - as values in candidates_outreach.l2_district_type
    - as actual columns in deid_voters

    These become the district columns we unpivot dynamically.
    """
    outreach_district_types = _get_outreach_district_types(spark)
    voter_columns = _get_voter_columns(spark)

    matches = [d for d in outreach_district_types if d in voter_columns]
    return sorted(matches)


# ============================================================================
# Helpers: district feature query construction
# ============================================================================

def _build_stack_expression(matching_district_cols: list[str]) -> str:
    """
    Build a Spark SQL stack() expression that converts wide district columns
    in deid_voters into a long two-column representation:
        (district_type, district_name)

    Example output:
        stack(
            3,
            'city', `city`,
            'county', `county`,
            'college_board_district', `college_board_district`
        ) AS (district_type, district_name)
    """
    if not matching_district_cols:
        raise ValueError("matching_district_cols must not be empty.")

    stack_pairs = ",\n            ".join(
        [f"'{col}', `{col}`" for col in matching_district_cols]
    )

    return f"""
        stack(
            {len(matching_district_cols)},
            {stack_pairs}
        ) AS (district_type, district_name)
    """


def _build_district_features_query(matching_district_cols: list[str]) -> str:
    """
    Build the SQL query that:
    1. unpivots relevant voter district columns into long format
    2. aggregates age / gender / education by district_type + district_name

    Returns
    -------
    str
        SQL query that produces one row per district with district-level
        demographic features.
    """
    if not matching_district_cols:
        raise ValueError("matching_district_cols must not be empty.")

    stack_expr = _build_stack_expression(matching_district_cols)

    # Education buckets may need adjustment if your source categories differ.
    return f"""
    WITH voter_district_long AS (
        SELECT
            LOWER(TRIM(district_type)) AS district_type,
            LOWER(TRIM(district_name)) AS district_name,
            age_int,
            gender,
            education_of_person
        FROM (
            SELECT
                {stack_expr},
                age_int,
                gender,
                education_of_person
            FROM {VOTER_TABLE}
        ) t
        WHERE district_name IS NOT NULL
          AND TRIM(district_name) <> ''
    )

    SELECT
        district_type,
        district_name,

        COUNT(*) AS district_n_voters,

        AVG(CAST(age_int AS DOUBLE)) AS avg_age_in_district,

        AVG(
            CASE
                WHEN UPPER(TRIM(gender)) = 'F' THEN 1.0
                ELSE 0.0
            END
        ) AS pct_female_in_district,

        AVG(
            CASE
                WHEN UPPER(TRIM(gender)) = 'M' THEN 1.0
                ELSE 0.0
            END
        ) AS pct_male_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Did Not Complete High School Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_no_hs_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Completed High School Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_hs_grad_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Attended But Did Not Complete College Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_some_college_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Attended Vocational/Technical School Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_vocational_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Completed College Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_college_grad_in_district,

        AVG(
            CASE
                WHEN education_of_person = 'Completed Graduate School Likely'
                THEN 1.0 ELSE 0.0
            END
        ) AS pct_grad_school_in_district

            FROM voter_district_long
            GROUP BY district_type, district_name
        """


def _build_enriched_outreach_query(base_query: str, matching_district_cols: list[str]) -> str:
    """
    Build the full outreach query with a LEFT JOIN to district demographic features.

    Parameters
    ----------
    base_query : str
        Base outreach query (training or scoring).
    matching_district_cols : list[str]
        District columns that exist in both outreach district types and the
        deid_voters schema.

    Returns
    -------
    str
        SQL query that returns outreach rows enriched with district demographic
        aggregates.
    """
    district_features_query = _build_district_features_query(matching_district_cols)

    return f"""
    WITH base_outreach AS (
        {base_query}
    ),
    district_demo_features AS (
        {district_features_query}
    )

    SELECT
        bo.*,
        ddf.district_n_voters,
        ddf.avg_age_in_district,
        ddf.pct_female_in_district,
        ddf.pct_male_in_district,
        ddf.pct_no_hs_in_district,
        ddf.pct_hs_grad_in_district,
        ddf.pct_some_college_in_district,
        ddf.pct_vocational_in_district,
        ddf.pct_college_grad_in_district,
        ddf.pct_grad_school_in_district
    FROM base_outreach bo
    LEFT JOIN district_demo_features ddf
        ON LOWER(TRIM(bo.l2_district_type)) = ddf.district_type
    AND LOWER(TRIM(bo.l2_district_name)) = ddf.district_name
    """


def _add_empty_district_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add district feature columns filled with missing values.

    This is used as a fallback when no matching district columns are found in
    deid_voters, so downstream code still sees a stable schema.
    """
    df = df.copy()

    empty_cols = [
        "district_n_voters",
        "avg_age_in_district",
        "pct_female_in_district",
        "pct_male_in_district",
        "pct_no_hs_in_district",
        "pct_hs_grad_in_district",
        "pct_some_college_in_district",
        "pct_vocational_in_district",
        "pct_college_grad_in_district",
        "pct_grad_school_in_district",
    ]

    for col in empty_cols:
        df[col] = pd.NA

    return df

def _coerce_district_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    district_numeric_cols = [
        "district_n_voters",
        "avg_age_in_district",
        "pct_female_in_district",
        "pct_male_in_district",
        "pct_no_hs_in_district",
        "pct_hs_grad_in_district",
        "pct_some_college_in_district",
        "pct_vocational_in_district",
        "pct_college_grad_in_district",
        "pct_grad_school_in_district",
    ]

    for col in district_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================================
# Public loaders
# ============================================================================

def load_training_data(spark) -> pd.DataFrame:
    """
    Load the training dataset from Spark SQL into pandas, enriched with
    district-level demographic features.

    Workflow
    --------
    1. Inspect outreach district types
    2. Match those values to real deid_voters columns
    3. Dynamically build district aggregates for age / gender / education
    4. Join those aggregates onto the outreach rows
    5. Return a pandas DataFrame

    Fallback behavior
    -----------------
    If no matching district columns are found in deid_voters, the function
    returns the base outreach rows with district feature columns added as nulls.
    """
    matching_district_cols = _get_matching_district_columns(spark)

    if not matching_district_cols:
        print("No matching district columns found in deid_voters. Returning base training data.")
        base_df = spark.sql(TRAINING_MESSAGE_QUERY).toPandas()
        return _add_empty_district_feature_columns(base_df)

    print(f"Using {len(matching_district_cols)} district columns for training data enrichment.")
    print(f"First few matching columns: {matching_district_cols[:10]}")

    query = _build_enriched_outreach_query(
        base_query=TRAINING_MESSAGE_QUERY,
        matching_district_cols=matching_district_cols,
    )
    df = spark.sql(query).toPandas()
    df = _coerce_district_numeric_cols(df)
    return df



def load_scoring_data(spark) -> pd.DataFrame:
    """
    Load the scoring dataset from Spark SQL into pandas, enriched with
    district-level demographic features.

    Uses the same district-matching and aggregation logic as training so the
    scoring schema stays aligned with the training schema.
    """
    matching_district_cols = _get_matching_district_columns(spark)

    if not matching_district_cols:
        print("No matching district columns found in deid_voters. Returning base scoring data.")
        base_df = spark.sql(SCORING_MESSAGE_QUERY).toPandas()
        return _add_empty_district_feature_columns(base_df)

    print(f"Using {len(matching_district_cols)} district columns for scoring data enrichment.")
    print(f"First few matching columns: {matching_district_cols[:10]}")

    query = _build_enriched_outreach_query(
        base_query=SCORING_MESSAGE_QUERY,
        matching_district_cols=matching_district_cols,
    )


    df = spark.sql(query).toPandas()
    df = _coerce_district_numeric_cols(df)
    
    return df


# ============================================================================
# Optional debug helpers
# ============================================================================


def preview_matching_district_columns(spark) -> list[str]:
    """
    Convenience helper for notebook debugging.

    Returns the matched district columns that will be used to build the
    district-level demographic features.
    """
    cols = _get_matching_district_columns(spark)
    print(f"{len(cols)} matching district columns found.")
    print(cols[:50])
    return cols


def preview_district_feature_query(spark) -> str:
    """
    Convenience helper for notebook debugging.

    Returns the generated district-feature SQL without executing it.
    """
    cols = _get_matching_district_columns(spark)
    if not cols:
        raise ValueError("No matching district columns found.")

    query = _build_district_features_query(cols)
    print(query)
    return query