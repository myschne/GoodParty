"""
SQL query definitions for candidate-level training and scoring datasets.

This module stores the Spark SQL used to build the model input tables at the
candidate-election level. It defines separate queries for training data and
scoring data, while keeping their feature structure aligned so the same
downstream feature engineering and modeling pipeline can be used consistently
across both training and inference.
"""


# =========================================================
# Candidate-level training dataset query
# =========================================================
# Purpose:
# Build one row per candidate-election combination for model training.
#
# Output grain:
# One row per (hubspot_id, election_date)
#
# Includes:
# - Static candidate / race attributes
# - Aggregated outreach history
# - Summary viability features
# - Most common outreach type
# - Binary target variable (Win)
#
# Notes:
# - Only includes races with known general election outcomes
# - Excludes uncontested races
# - latest_outreach only considers outreach on or before election_date
# =========================================================

TRAINING_QUERY = """
WITH base AS (
    SELECT
        -- Candidate identifier, standardized as trimmed string
        TRIM(CAST(hubspot_id AS STRING)) AS hubspot_id,

        -- Election / geography / office fields
        election_date,
        state,
        office_level,
        office_type,

        -- Original viability score recorded at outreach row level
        viability_score,

        -- Raw election outcome text
        general_election_result,

        -- Outreach activity details
        outreach_date,
        outreach_type,

        -- Race structure / candidate context
        number_of_opponents,
        partisan_type,
        seats_available,

        -- Convert text flags to binary indicators
        CASE WHEN is_open_seat = 'Yes' THEN 1 ELSE 0 END AS open_seat,
        CASE WHEN is_incumbent = 'Yes' THEN 1 ELSE 0 END AS incumbent,

        -- Binary training target:
        -- 1 if candidate won the general election, else 0
        CASE WHEN general_election_result = 'Won General' THEN 1 ELSE 0 END AS Win

    FROM mart_mban2026.candidates_outreach

    -- Keep only observations with known outcomes
    -- and remove uncontested races
    WHERE general_election_result IS NOT NULL
      AND (is_uncontested IS NULL OR is_uncontested <> 'UnContested')
),

cand AS (
    SELECT
        -- Candidate-election grain
        hubspot_id,
        election_date,

        -- Carry forward candidate / race attributes
        -- FIRST(..., true) returns first non-null value in Spark SQL
        FIRST(state, true) AS state,
        FIRST(office_level, true) AS office_level,
        FIRST(office_type, true) AS office_type,
        FIRST(open_seat, true) AS open_seat,
        FIRST(incumbent, true) AS incumbent,
        FIRST(number_of_opponents, true) AS number_of_opponents,
        FIRST(partisan_type, true) AS partisan_type,

        -- Seat structure: take max in case repeated across rows
        MAX(seats_available) AS seats_available,

        -- Outreach intensity feature:
        -- total number of outreach records for this candidate-election
        COUNT(*) AS n_outreach_rows,

        -- Summary statistics of viability score across outreach rows
        AVG(viability_score) AS viability_score_mean,
        MAX(viability_score) AS viability_score_max,

        -- Most recent outreach date on or before the election
        MAX(CASE WHEN outreach_date <= election_date THEN outreach_date END) AS latest_outreach,

        -- Candidate-level target:
        -- max works because Win is binary and should be consistent within group
        MAX(Win) AS Win

    FROM base
    GROUP BY hubspot_id, election_date
),

most_common_type AS (
    SELECT hubspot_id, election_date, outreach_type
    FROM (
        SELECT
            hubspot_id,
            election_date,
            outreach_type,

            -- Rank outreach types by frequency within each candidate-election group
            -- Tie-break alphabetically by outreach_type
            ROW_NUMBER() OVER (
                PARTITION BY hubspot_id, election_date
                ORDER BY COUNT(*) DESC, outreach_type
            ) AS rn
        FROM base

        -- Ignore null / blank outreach types
        WHERE outreach_type IS NOT NULL AND LENGTH(TRIM(outreach_type)) > 0
        GROUP BY hubspot_id, election_date, outreach_type
    )

    -- Keep only the top-ranked outreach type per candidate-election
    WHERE rn = 1
)

SELECT
    -- Candidate-level aggregated features
    c.*,

    -- Add most common outreach type as categorical feature
    mct.outreach_type AS most_common_outreach_type

FROM cand c
LEFT JOIN most_common_type mct
    ON c.hubspot_id = mct.hubspot_id
   AND c.election_date = mct.election_date
"""


# =========================================================
# Candidate-level scoring dataset query
# =========================================================
# Purpose:
# Build one row per candidate-election combination for scoring /
# future prediction.
#
# Output grain:
# One row per (hubspot_id, election_date)
#
# Includes:
# - Same feature structure as training query
# - No target variable (Win)
#
# Notes:
# - Keeps all rows from the source table
# - Does not require known election outcomes
# - latest_outreach only considers outreach on or before election_date
# =========================================================

SCORING_QUERY = """
WITH base AS (
    SELECT
        -- Candidate identifier, standardized as trimmed string
        TRIM(CAST(hubspot_id AS STRING)) AS hubspot_id,

        -- Election / geography / office fields
        election_date,
        state,
        office_level,
        office_type,

        -- Original viability score recorded at outreach row level
        viability_score,

        -- Outreach activity details
        outreach_date,
        outreach_type,

        -- Race structure / candidate context
        number_of_opponents,
        partisan_type,
        seats_available,

        -- Convert text flags to binary indicators
        CASE WHEN is_open_seat = 'Yes' THEN 1 ELSE 0 END AS open_seat,
        CASE WHEN is_incumbent = 'Yes' THEN 1 ELSE 0 END AS incumbent

    FROM mart_mban2026.candidates_outreach
),

cand AS (
    SELECT
        -- Candidate-election grain
        hubspot_id,
        election_date,

        -- Carry forward candidate / race attributes
        FIRST(state, true) AS state,
        FIRST(office_level, true) AS office_level,
        FIRST(office_type, true) AS office_type,
        FIRST(open_seat, true) AS open_seat,
        FIRST(incumbent, true) AS incumbent,
        FIRST(number_of_opponents, true) AS number_of_opponents,
        FIRST(partisan_type, true) AS partisan_type,

        -- Seat structure
        MAX(seats_available) AS seats_available,

        -- Outreach intensity feature
        COUNT(*) AS n_outreach_rows,

        -- Summary statistics of viability across outreach rows
        AVG(viability_score) AS viability_score_mean,
        MAX(viability_score) AS viability_score_max,

        -- Most recent outreach date on or before the election
        MAX(CASE WHEN outreach_date <= election_date THEN outreach_date END) AS latest_outreach

    FROM base
    GROUP BY hubspot_id, election_date
),

most_common_type AS (
    SELECT hubspot_id, election_date, outreach_type
    FROM (
        SELECT
            hubspot_id,
            election_date,
            outreach_type,

            -- Rank outreach types by frequency within each candidate-election group
            -- Tie-break alphabetically by outreach_type
            ROW_NUMBER() OVER (
                PARTITION BY hubspot_id, election_date
                ORDER BY COUNT(*) DESC, outreach_type
            ) AS rn
        FROM base

        -- Ignore null / blank outreach types
        WHERE outreach_type IS NOT NULL AND LENGTH(TRIM(outreach_type)) > 0
        GROUP BY hubspot_id, election_date, outreach_type
    )

    -- Keep only the top-ranked outreach type per candidate-election
    WHERE rn = 1
)

SELECT
    -- Candidate-level aggregated features
    c.*,

    -- Add most common outreach type as categorical feature
    mct.outreach_type AS most_common_outreach_type

FROM cand c
LEFT JOIN most_common_type mct
    ON c.hubspot_id = mct.hubspot_id
   AND c.election_date = mct.election_date
"""

TRAINING_MESSAGE_QUERY = """
SELECT
    TRIM(CAST(hubspot_id AS STRING)) AS hubspot_id,
    election_date,
    outreach_date,
    state,
    office_level,
    office_type,
    viability_score,
    outreach_type,
    script,
    number_of_opponents,
    partisan_type,
    seats_available,
    CASE WHEN is_open_seat = 'Yes' THEN 1 ELSE 0 END AS open_seat,
    CASE WHEN is_incumbent = 'Yes' THEN 1 ELSE 0 END AS incumbent,
    general_election_result,
    CASE WHEN general_election_result = 'Won General' THEN 1 ELSE 0 END AS Win
FROM mart_mban2026.candidates_outreach
WHERE general_election_result IS NOT NULL
  AND (is_uncontested IS NULL OR is_uncontested <> 'UnContested')
"""

SCORING_MESSAGE_QUERY = """
SELECT
    TRIM(CAST(hubspot_id AS STRING)) AS hubspot_id,
    election_date,
    outreach_date,
    state,
    office_level,
    office_type,
    viability_score,
    outreach_type,
    script,
    number_of_opponents,
    partisan_type,
    seats_available,
    CASE WHEN is_open_seat = 'Yes' THEN 1 ELSE 0 END AS open_seat,
    CASE WHEN is_incumbent = 'Yes' THEN 1 ELSE 0 END AS incumbent
FROM mart_mban2026.candidates_outreach
WHERE (is_uncontested IS NULL OR is_uncontested <> 'UnContested')
"""

