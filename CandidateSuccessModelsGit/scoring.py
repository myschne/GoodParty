"""
Scoring utilities for applying a trained classifier to candidate data.

This module contains helper logic for inference-time scoring. Its main job is to:
1. Prepare raw scoring data using the same feature engineering logic as training
2. Generate predicted win probabilities from a trained model
3. Convert probabilities into binary win predictions using the configured threshold
4. Map probabilities into the project's viability score and viability bucket system
5. Return a compact candidate-level output table for downstream use

This keeps prediction logic separate from the main scoring script and ensures
that feature preparation and output formatting are reusable.
"""
from config import THRESHOLD
from feature_engineering import (
    split_X_y,
    proba_to_viability_score,
    viability_score_to_bucket,
)


def score_candidates(model, score_df):
    """
    Score candidate rows with a trained model and return a compact output table.

    Workflow
    --------
    1. Apply feature engineering via `split_X_y`
    2. Generate predicted probabilities for class 1 (Win)
    3. Convert probabilities to binary predictions using THRESHOLD
    4. Convert probabilities to the project's viability score
    5. Convert viability scores to labeled viability buckets
    6. Return a selected set of identifying and prediction-related columns

    Parameters
    ----------
    model : sklearn-like estimator
        Trained model object with a `predict_proba` method.
    score_df : pandas.DataFrame
        Raw candidate data to score. Must contain the columns needed by
        `split_X_y`, including any identifier and metadata columns that
        should remain in the final output.

    Returns
    -------
    pandas.DataFrame
        Candidate-level scored output containing:
        - hubspot_id
        - election_date
        - state
        - office_level
        - win_probability
        - predicted_win
        - viability_score_model
        - viability_bucket_model
    """
    # Prepare features using the same feature engineering logic used in training.
    # A placeholder target column is expected by split_X_y, even though the goal
    # here is scoring rather than supervised training evaluation.
    X_score, _, featured_df = split_X_y(score_df, target_col="Win")

    # Predict probability of the positive class: Win = 1.
    proba = model.predict_proba(X_score)[:, 1]

    # Convert probabilities to binary predictions using the configured threshold.
    pred = (proba >= THRESHOLD).astype(int)

    # Start from the engineered candidate-level frame so key metadata columns
    # remain aligned with the generated predictions.
    scored = featured_df.copy()
    scored["win_probability"] = proba
    scored["predicted_win"] = pred

    # Map probabilities into the project's 0-5 viability score scale, then
    # into the corresponding labeled viability bucket.
    scored["viability_score_model"] = proba_to_viability_score(proba)
    scored["viability_bucket_model"] = viability_score_to_bucket(
        scored["viability_score_model"]
    )

    # Return only the core candidate identifiers and model outputs needed for
    # downstream review, export, or table storage.
    return scored[
        [
            "hubspot_id",
            "election_date",
            "state",
            "office_level",
            "win_probability",
            "predicted_win",
            "viability_score_model",
            "viability_bucket_model",
        ]
    ]