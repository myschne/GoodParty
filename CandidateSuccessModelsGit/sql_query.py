"""
SQL query definitions for candidate-level training and scoring datasets.

This module stores the Spark SQL used to build the model input tables at the
candidate-election level. It defines separate queries for training data and
scoring data, while keeping their feature structure aligned so the same
downstream feature engineering and modeling pipeline can be used consistently
across both training and inference.
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
    is_partisan,
    seats_available,
    CASE WHEN is_open_seat THEN 1 ELSE 0 END AS open_seat,
    CASE WHEN is_incumbent THEN 1 ELSE 0 END AS incumbent,
    general_election_result,
    CASE WHEN general_election_result = 'Won General' THEN 1 ELSE 0 END AS Win
FROM mart_mban2026.candidates_outreach
WHERE general_election_result IS NOT NULL
  AND (is_uncontested IS NULL OR is_uncontested = FALSE)
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
    is_partisan,
    seats_available,
    CASE WHEN is_open_seat THEN 1 ELSE 0 END AS open_seat,
    CASE WHEN is_incumbent THEN 1 ELSE 0 END AS incumbent
FROM mart_mban2026.candidates_outreach
WHERE (is_uncontested IS NULL OR is_uncontested = FALSE)
"""