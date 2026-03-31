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
import pandas as pd

from config import THRESHOLD
from feature_engineering import (
    split_X_y,
    proba_to_viability_score,
    viability_score_to_bucket,
)


def score_candidates(model, score_df):
    """
    Score candidate-election rows with a trained model and return a compact
    prediction table for downstream use.

    This function applies the same feature-engineering pipeline used during
    training, generates predicted win probabilities, converts probabilities
    into binary win predictions, and maps outputs into the project's
    viability score / bucket system.

    It also applies a business-rule override for uncontested races:
    - win_probability = 1.0
    - predicted_win = 1
    """
    # Build the scoring feature matrix using the same feature engineering
    # logic as training. `featured_df` retains identifiers and metadata
    # needed in the final scored output.
    X_score, _, featured_df = split_X_y(score_df, target_col="Win")

    # Generate predicted probability of the positive class (Win = 1).
    proba = model.predict_proba(X_score)[:, 1]

    # Convert probabilities to binary predictions using the configured threshold.
    pred = (proba >= THRESHOLD).astype(int)

    # Start from the fully featured dataframe so key identifiers and metadata
    # remain aligned with model outputs.
    scored = featured_df.copy()
    scored["win_probability"] = proba
    scored["predicted_win"] = pred

    # Apply a scoring-time business rule for uncontested races:
    # candidates in uncontested elections are assigned a guaranteed win.
    if "is_uncontested" in scored.columns:
        uncontested_mask = (
            pd.to_numeric(scored["is_uncontested"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )

        scored.loc[uncontested_mask, "win_probability"] = 1.0
        scored.loc[uncontested_mask, "predicted_win"] = 1

    # Convert the final win probability into the project's 0-5 viability score.
    scored["viability_score_model"] = proba_to_viability_score(
        scored["win_probability"]
    )

    # Map the continuous viability score into labeled viability buckets.
    scored["viability_bucket_model"] = viability_score_to_bucket(
        scored["viability_score_model"]
    )

    # Return only the core identifiers and prediction outputs needed for
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