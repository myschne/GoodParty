"""
Cross-validation utilities for grouped model validation.

This module defines helper logic for constructing grouped cross-validation
folds. Grouped validation is important when multiple rows belong to the same
underlying candidate-election unit, because it prevents information from the
same group from appearing in both training and validation data.

In this project, groups are defined at the candidate-election level using:
    hubspot_id | election_date

This ensures that all observations for a given candidate in a given election
stay together in the same fold, reducing leakage and producing a more realistic
estimate of out-of-sample performance.
"""

from sklearn.model_selection import GroupKFold

def make_group_folds(df, n_splits=5) -> list[dict]:
    """
    Create grouped CV folds using candidate-election groups.

    Group definition:
        hubspot_id | election_date

    This keeps all rows for the same candidate-election combination
    in the same fold, preventing leakage across train/test.
    """
    group_id = (
        df["hubspot_id"].astype(str).str.strip() + "|" +
        df["election_date"].astype(str)
    )

    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(
        gkf.split(X=df, y=None, groups=group_id)
    ):
        folds.append({
            "fold": fold_id,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return folds