""" Evaluation and reporting utilities for model validation. This module contains helper functions used after model training to standardize prediction outputs, compute classification metrics, compare model-derived viability scores against the original benchmark scores, and aggregate feature importance across cross-validation folds. The functions in this file support several stages of the evaluation workflow: - organizing fold-level metrics into tidy tables - building row-level out-of-fold prediction frames - standardizing model outputs across estimator types - computing fold-level and pooled classification metrics - comparing model viability outputs to original viability labels - summarizing feature importance stability across folds Keeping these functions in one module makes the training pipeline cleaner and ensures that evaluation logic is reusable, consistent, and easier to maintain. 
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from feature_engineering import (
    proba_to_viability_score,
    viability_score_to_bucket,
)

# =========================================================
# Fold-level output helpers
# =========================================================

def build_fold_metrics_df(fold_metrics_list):
    """
    Convert a list of fold metric dictionaries into a tidy summary DataFrame.

    This is mainly a reporting helper used after cross-validation. It takes
    the raw list of per-fold metric outputs and returns a table where each
    row corresponds to one fold.

    Parameters
    ----------
    fold_metrics_list : list[dict]
        List of dictionaries returned by `compute_fold_metrics()`. Each
        dictionary is expected to contain:
        - fold
        - roc_auc
        - accuracy
        - precision
        - recall
        - f1_score
        - n_samples

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per fold and columns:
        - fold
        - roc_auc
        - accuracy
        - precision
        - recall
        - f1_score
        - n_samples
    """
    return pd.DataFrame([
        {
            "fold": m["fold"],
            "roc_auc": m["roc_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "n_samples": m["n_samples"],
        }
        for m in fold_metrics_list
    ])


def build_fold_prediction_frame(featured_test, y_true, pred_outputs, fold_id):
    """
    Build a row-level prediction DataFrame for one test fold.

    This function combines:
    - the fold's original feature/metadata rows (`featured_test`)
    - the true label (`y_true`)
    - standardized model prediction outputs (`pred_outputs`)
    - the fold identifier (`fold_id`)

    It is used to construct the out-of-fold prediction table that later
    supports pooled evaluation, viability comparison, plotting, and exports.

    Parameters
    ----------
    featured_test : pandas.DataFrame
        Test-fold dataset retaining the row-level columns you want in the
        final output table, such as identifiers, metadata, or engineered
        features.
    y_true : array-like
        True labels for the test fold.
    pred_outputs : dict
        Standardized prediction dictionary returned by
        `get_model_predictions()`. Expected keys:
        - pred_label
        - pred_score
        - pred_proba
        - prediction_type
    fold_id : int or str
        Fold identifier to attach to each row.

    Returns
    -------
    pandas.DataFrame
        Row-level test-fold prediction table containing:
        - original `featured_test` columns
        - fold
        - y_true
        - pred_label
        - prediction_type
        - pred_score (if available)
        - pred_proba (if available)
    """
    df = featured_test.copy()
    df["fold"] = fold_id
    df["y_true"] = np.asarray(y_true)

    df["pred_label"] = pred_outputs["pred_label"]
    df["prediction_type"] = pred_outputs["prediction_type"]

    if pred_outputs["pred_score"] is not None:
        df["pred_score"] = pred_outputs["pred_score"]

    if pred_outputs["pred_proba"] is not None:
        df["pred_proba"] = pred_outputs["pred_proba"]

    return df


def get_model_predictions(clf, X, threshold=0.5):
    """
    Return model outputs in a standardized format regardless of estimator type.

    Different sklearn estimators expose predictions in different ways:
    - probabilistic classifiers may support `predict_proba`
    - margin-based classifiers may support `decision_function`
    - some estimators only support `predict`

    This helper normalizes those interfaces into one dictionary so downstream
    evaluation code does not need model-specific branches.

    Parameters
    ----------
    clf : fitted estimator
        Trained model or sklearn pipeline.
    X : array-like or pandas.DataFrame
        Feature matrix to score.
    threshold : float, default=0.5
        Probability threshold used to convert predicted probabilities into
        binary labels when `predict_proba` is available.

    Returns
    -------
    dict
        Dictionary with keys:
        - pred_label : numpy.ndarray
            Binary or label predictions.
        - pred_score : numpy.ndarray or None
            Continuous score used for ranking metrics like ROC-AUC. This is
            usually probability of class 1 or decision-function score.
        - pred_proba : numpy.ndarray or None
            Predicted probability of class 1 when available.
        - prediction_type : str
            One of:
            - "probability"
            - "decision_function"
            - "label_only"

    Notes
    -----
    Behavior by estimator interface:
    - If `predict_proba` exists:
        - uses class-1 probability as both `pred_proba` and `pred_score`
        - applies `threshold` to create `pred_label`
    - Else if `decision_function` exists:
        - stores decision scores in `pred_score`
        - uses `predict()` for `pred_label`
    - Else:
        - only stores `pred_label`
    """
    outputs = {
        "pred_label": None,
        "pred_score": None,
        "pred_proba": None,
        "prediction_type": None,
    }

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            proba_1 = proba[:, 1]
        else:
            proba_1 = proba.ravel()

        outputs["pred_proba"] = proba_1
        outputs["pred_score"] = proba_1
        outputs["pred_label"] = (proba_1 >= threshold).astype(int)
        outputs["prediction_type"] = "probability"

    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X)
        outputs["pred_score"] = np.asarray(score).ravel()
        outputs["pred_label"] = clf.predict(X)
        outputs["prediction_type"] = "decision_function"

    else:
        pred = clf.predict(X)
        outputs["pred_label"] = np.asarray(pred).ravel()
        outputs["prediction_type"] = "label_only"

    return outputs


# =========================================================
# Classification metric helpers
# =========================================================

def compute_fold_metrics(y_true, pred_label, pred_score=None, fold_id=None):
    """
    Compute evaluation metrics for a single cross-validation fold.

    This function evaluates one fold's predictions and returns both
    threshold-based classification metrics and, when available, ranking-based
    ROC-AUC.

    Parameters
    ----------
    y_true : array-like
        True binary labels for the fold.
    pred_label : array-like
        Predicted binary labels for the fold.
    pred_score : array-like, optional
        Continuous model scores for the fold, usually predicted probabilities
        or decision-function values. Used to compute ROC-AUC when available.
    fold_id : int or str, optional
        Fold identifier to store with the output metrics.

    Returns
    -------
    dict
        Dictionary containing:
        - fold : fold identifier
        - n_samples : number of observations in the fold
        - accuracy : classification accuracy
        - precision : positive-class precision
        - recall : positive-class recall
        - f1_score : harmonic mean of precision and recall
        - confusion_matrix : 2x2 confusion matrix
        - classification_report : sklearn text summary report
        - roc_auc : ROC-AUC based on `pred_score`, or NaN if unavailable

    Notes
    -----
    ROC-AUC requires:
    - a continuous score vector
    - at least two observed classes in `y_true`

    If ROC-AUC cannot be computed, the function returns `np.nan` instead of
    raising an exception.
    """
    y_true = np.asarray(y_true)
    pred_label = np.asarray(pred_label)

    metrics = {
        "fold": fold_id,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1_score": f1_score(y_true, pred_label, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred_label),
        "classification_report": classification_report(
            y_true, pred_label, zero_division=0
        ),
    }

    # ROC-AUC requires continuous scores and at least two classes in y_true.
    if pred_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, pred_score)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def compute_pooled_metrics(y_true, pred_label, pred_score=None):
    """
    Compute evaluation metrics on pooled out-of-fold predictions.

    This is typically used after concatenating predictions from all
    cross-validation folds into one combined out-of-fold dataset. It provides
    a single overall performance summary using all OOF rows together.

    Parameters
    ----------
    y_true : array-like
        True binary labels across all pooled observations.
    pred_label : array-like
        Predicted binary labels across all pooled observations.
    pred_score : array-like, optional
        Continuous model scores across all pooled observations, used to
        compute ROC-AUC when available.

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy : pooled classification accuracy
        - precision : pooled positive-class precision
        - recall : pooled positive-class recall
        - f1_score : pooled F1 score
        - confusion_matrix : pooled confusion matrix
        - classification_report : pooled sklearn text summary report
        - roc_auc : pooled ROC-AUC, or NaN if unavailable

    Notes
    -----
    This function is structurally similar to `compute_fold_metrics()`, but it
    does not include `fold` or `n_samples` because it summarizes the full
    pooled out-of-fold prediction set rather than one fold.
    """
    y_true = np.asarray(y_true)
    pred_label = np.asarray(pred_label)

    metrics = {
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1_score": f1_score(y_true, pred_label, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred_label),
        "classification_report": classification_report(
            y_true, pred_label, zero_division=0
        ),
    }

    # ROC-AUC requires continuous scores and at least two classes in y_true.
    if pred_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, pred_score)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


# =========================================================
# Viability comparison helpers
# =========================================================

def compare_model_vs_original_viability(results):
    """
    Compare original viability scores against model-derived viability scores.

    This function creates a model-based viability score and bucket, aligns it
    with the original benchmark viability score, and computes multiple
    agreement diagnostics.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing either:
        - `viability_score_model`, or
        - `proba_win` so that `viability_score_model` can be derived

        It should also contain:
        - `viability_score_mean` for the original viability benchmark

    Returns
    -------
    dict
        Dictionary containing:
        - results : pandas.DataFrame
            DataFrame with added comparison columns.
        - crosstab : pandas.DataFrame
            Contingency table of original vs model viability buckets.
        - exact_agreement : float
            Share of rows with identical original and model buckets.
        - bucket_distance_distribution : pandas.Series
            Distribution of absolute bucket distance.
        - mean_bucket_distance : float
            Average absolute bucket distance.
        - score_correlation : float
            Correlation between original and model viability scores.

    Raises
    ------
    KeyError
        If neither `viability_score_model` nor `proba_win` is present.

    Notes
    -----
    Added columns include:
    - viability_score_model
    - viability_bucket_model
    - viability_bucket_num
    - viability_bucket_orig
    - viability_bucket_orig_num
    - bucket_distance

    Workflow
    --------
    1. Create `viability_score_model` if it does not exist.
    2. Convert model viability score into ordered bucket labels.
    3. Convert original average viability score into ordered bucket labels.
    4. Compare original and model bucket positions.
    5. Return agreement summaries and the enriched results table.
    """
    results = results.copy()

    # Create model-derived viability columns if they do not already exist.
    if "viability_score_model" not in results.columns:
        if "proba_win" not in results.columns:
            raise KeyError(
                "compare_model_vs_original_viability requires either "
                "'viability_score_model' or 'proba_win' in results."
            )
        results["viability_score_model"] = proba_to_viability_score(results["proba_win"])

    if "viability_bucket_model" not in results.columns:
        results["viability_bucket_model"] = viability_score_to_bucket(
            results["viability_score_model"]
        )

    if "viability_bucket_num" not in results.columns:
        results["viability_bucket_num"] = (
            results["viability_bucket_model"].cat.codes + 1
        )

    # Convert the original average viability score into ordered bucket labels.
    results["viability_bucket_orig"] = viability_score_to_bucket(
        pd.to_numeric(results["viability_score_mean"], errors="coerce")
    )
    results["viability_bucket_orig_num"] = (
        results["viability_bucket_orig"].cat.codes + 1
    )

    # Measure how far apart the original and model buckets are.
    results["bucket_distance"] = (
        results["viability_bucket_orig_num"] - results["viability_bucket_num"]
    ).abs()

    # Correlation is computed only on rows where both score values are present.
    valid = results[["viability_score_mean", "viability_score_model"]].dropna()
    corr = (
        valid["viability_score_mean"].corr(valid["viability_score_model"])
        if len(valid) > 1 else np.nan
    )

    return {
        "results": results,
        "crosstab": pd.crosstab(
            results["viability_bucket_orig"],
            results["viability_bucket_model"],
            dropna=False,
        ),
        "exact_agreement": float(
            (results["viability_bucket_orig"] == results["viability_bucket_model"]).mean()
        ),
        "bucket_distance_distribution": (
            results["bucket_distance"].value_counts(dropna=False).sort_index()
        ),
        "mean_bucket_distance": float(results["bucket_distance"].mean()),
        "score_correlation": float(corr) if pd.notna(corr) else np.nan,
    }


# =========================================================
# Feature importance aggregation
# =========================================================

def aggregate_feature_importance(importance_series_list, signed=True):
    """
    Aggregate feature importance or coefficient values across CV folds.

    Parameters
    ----------
    importance_series_list : list[pandas.Series]
        List of per-fold feature importance series.
    signed : bool, default=True
        Whether the values have directional meaning (e.g. linear coefficients).
        If False, sign_consistency is returned as NaN.
    """
    if not importance_series_list:
        return pd.DataFrame(
            columns=["mean_importance", "std_importance", "mean_abs_importance", "sign_consistency"]
        )

    imp_df = pd.concat(importance_series_list, axis=1).fillna(0.0)

    out = pd.DataFrame({
        "mean_importance": imp_df.mean(axis=1),
        "std_importance": imp_df.std(axis=1),
        "mean_abs_importance": imp_df.abs().mean(axis=1),
    })

    if signed:
        out["sign_consistency"] = (
            np.sign(imp_df).replace(0, np.nan).mean(axis=1).abs()
        )
    else:
        out["sign_consistency"] = np.nan

    return out.sort_values("mean_abs_importance", ascending=False)