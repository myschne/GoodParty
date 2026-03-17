# LogisticModelViz.py
"""
Visualization utilities for evaluating binary classification model outputs.

This module is designed to work with outputs from the training/evaluation
pipeline, especially:
- fold-level performance metrics
- out-of-fold (OOF) predictions
- final candidate-level results
- aggregated feature importance summaries

Primary use cases:
1. Plot pooled prediction diagnostics
2. Plot confusion matrices and threshold behavior
3. Compare model-derived viability buckets to original buckets
4. Visualize feature importance in several interpretable ways

Expected inputs:
- fold_metrics_df: DataFrame with one row per CV fold and metrics such as roc_auc
- oof_df: DataFrame containing out-of-fold predictions and true labels
- results: candidate-level DataFrame with model probabilities, outcomes, and buckets
- imp: feature importance DataFrame aggregated across folds

This file does not train models. It only visualizes outputs that were already
computed elsewhere.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


# =========================================================
# Small utilities
# =========================================================

def _safe_pred_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert predicted probabilities into binary class predictions.

    Parameters
    ----------
    proba : np.ndarray
        Array of predicted probabilities for the positive class.
    threshold : float
        Classification threshold. Values >= threshold are labeled 1,
        otherwise 0.

    Returns
    -------
    np.ndarray
        Binary prediction array of 0s and 1s.
    """
    return (proba >= threshold).astype(int)


def _confusion_rates(cm: np.ndarray):
    """
    Compute standard rates from a 2x2 confusion matrix.

    Assumes the confusion matrix follows sklearn's convention:
        [[tn, fp],
         [fn, tp]]

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix.

    Returns
    -------
    dict
        Dictionary containing:
        - TPR: true positive rate / recall / sensitivity
        - FPR: false positive rate
        - TNR: true negative rate / specificity
        - FNR: false negative rate
    """
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    return {"TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr}


def pretty_feature_name(name: str) -> str:
    """
    Convert a raw engineered feature name into a cleaner, presentation-friendly label.

    This is mainly used for charts so axis labels are easier to interpret.

    Parameters
    ----------
    name : str
        Raw feature name from the model matrix or transformed feature set.

    Returns
    -------
    str
        Human-readable feature label.
    """
    s = str(name)

    s = s.replace("state_usps_", "State: ")
    s = s.replace("region_", "Region: ")
    s = s.replace("office_level_clean_", "Office level: ")
    s = s.replace("office_type_", "Office type: ")
    s = s.replace("partisan_type_", "Partisan: ")
    s = s.replace("incumbency_status_", "Incumbency: ")
    s = s.replace("election_dow_", "Election DOW: ")

    s = s.replace("candidates - available seats", "Competitiveness (candidates − seats)")
    s = s.replace("candidates_minus_available_seats", "Competitiveness (candidates − seats)")
    s = s.replace("days_between_outreach_and_election", "Days between outreach & election")
    s = s.replace("n_outreach_rows", "# outreach rows")

    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def feature_domain(name: str) -> str:
    """
    Assign a feature to a broad interpretive domain.

    This is used for grouped feature importance plots so related variables
    can be shown together.

    Parameters
    ----------
    name : str
        Raw feature name.

    Returns
    -------
    str
        One of:
        - Geography
        - Office
        - Race structure
        - Election timing
        - Outreach
        - Election Type
    """
    s = str(name)
    if s.startswith("state_usps_") or s.startswith("region_"):
        return "Geography"
    if s.startswith("office_level_clean_") or s.startswith("office_type_"):
        return "Office"
    if s.startswith("incumbency_status_") or "seats" in s or "opponents" in s:
        return "Race structure"
    if "election_" in s or "midterm" in s or "presidential" in s or "normal_election" in s:
        return "Election timing"
    if "outreach" in s:
        return "Outreach"
    return "Election Type"


# =========================================================
# Plotting functions
# =========================================================

def plot_score_hist_by_class(y_true, proba, threshold, savepath=None):
    """
    Plot predicted probability distributions separately for each true class.

    This helps assess class separation. Ideally, the positive and negative
    classes should have visibly different probability distributions.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    proba : array-like
        Predicted probabilities for class 1.
    threshold : float
        Classification threshold to mark on the histogram.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure()
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)

    plt.hist(proba[y_true == 0], bins=40, alpha=0.7, label="Win=0")
    plt.hist(proba[y_true == 1], bins=40, alpha=0.7, label="Win=1")
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold}")
    plt.xlabel("Predicted P(Win=1)")
    plt.ylabel("Count")
    plt.title("Predicted probability distribution by true class")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_confusion_matrix(y_true, proba, threshold, normalize=False, savepath=None):
    """
    Plot a confusion matrix based on a probability threshold.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    proba : array-like
        Predicted probabilities for class 1.
    threshold : float
        Threshold used to convert probabilities to class predictions.
    normalize : bool, default=False
        If True, row-normalize the matrix so each row sums to 1.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    tuple
        (fig, cm, rates) where:
        - fig is the matplotlib figure
        - cm is the raw 2x2 confusion matrix
        - rates is a dictionary of derived rates from _confusion_rates()
    """
    fig = plt.figure()
    pred = _safe_pred_from_proba(proba, threshold)
    cm = confusion_matrix(y_true, pred)
    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm_display,
            row_sums,
            out=np.zeros_like(cm_display),
            where=row_sums != 0,
        )

    plt.imshow(cm_display, aspect="auto")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))

    for (i, j), v in np.ndenumerate(cm_display):
        txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig, cm, _confusion_rates(cm)


def plot_threshold_sweep(y_true, proba, savepath=None):
    """
    Plot how classification metrics change across thresholds.

    Metrics included:
    - precision
    - recall
    - F1
    - TPR
    - FPR

    This is useful for selecting a threshold based on business tradeoffs.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    proba : array-like
        Predicted probabilities for class 1.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s, tprs, fprs = [], [], [], [], []

    for t in thresholds:
        pred = _safe_pred_from_proba(proba, t)
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
        cm = confusion_matrix(y_true, pred)
        rates = _confusion_rates(cm)
        tprs.append(rates["TPR"])
        fprs.append(rates["FPR"])

    fig = plt.figure()
    plt.plot(thresholds, precisions, label="precision")
    plt.plot(thresholds, recalls, label="recall")
    plt.plot(thresholds, f1s, label="f1")
    plt.plot(thresholds, tprs, label="TPR (recall)", linestyle="--")
    plt.plot(thresholds, fprs, label="FPR", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold sweep (pooled)")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_viability_bucket_distributions(results: pd.DataFrame, savepath=None):
    """
    Compare the distribution of original viability buckets vs model buckets.

    Parameters
    ----------
    results : pd.DataFrame
        Candidate-level results containing:
        - viability_bucket_orig
        - viability_bucket_model
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure()

    order = [
        "No Chance",
        "Unlikely to Win",
        "Has a Chance",
        "Likely to Win",
        "Frontrunner",
    ]

    orig = results["viability_bucket_orig"].astype(str).value_counts()
    model = results["viability_bucket_model"].astype(str).value_counts()

    orig = orig.reindex(order, fill_value=0)
    model = model.reindex(order, fill_value=0)

    x = np.arange(len(order))
    width = 0.4

    plt.bar(x - width / 2, orig.values, width=width, label="Original")
    plt.bar(x + width / 2, model.values, width=width, label="Model")
    plt.xticks(x, order, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Viability bucket distribution: Original vs Model")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_bucket_distance(results: pd.DataFrame, savepath=None):
    """
    Plot the distribution of bucket distance between original and model buckets.

    A distance of:
    - 0 means exact match
    - 1 means off by one bucket
    - 2+ means larger disagreement

    Parameters
    ----------
    results : pd.DataFrame
        Candidate-level results containing a bucket_distance column.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure()
    dist = results["bucket_distance"].value_counts(dropna=False).sort_index()

    plt.bar(dist.index.astype(str), dist.values)
    plt.xlabel("Bucket distance (0=match)")
    plt.ylabel("Count")
    plt.title("Bucket distance distribution")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_bucket_crosstab_heatmap(results: pd.DataFrame, savepath=None):
    """
    Plot a row-normalized heatmap comparing original vs model viability buckets.

    Each row represents an original bucket, and values show the share assigned
    by the model to each predicted bucket.

    Parameters
    ----------
    results : pd.DataFrame
        Candidate-level results containing:
        - viability_bucket_orig
        - viability_bucket_model
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.8))

    ct = pd.crosstab(
        results["viability_bucket_orig"],
        results["viability_bucket_model"],
        dropna=False,
    )

    row_sums = ct.sum(axis=1).replace(0, np.nan)
    pct = ct.div(row_sums, axis=0)

    vmax = float(np.nanmax(pct.values)) if pct.size else 1.0
    im = ax.imshow(pct.values, aspect="auto", cmap="RdBu", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(pct.shape[1]))
    ax.set_xticklabels(pct.columns.astype(str), rotation=30, ha="right")
    ax.set_yticks(np.arange(pct.shape[0]))
    ax.set_yticklabels(pct.index.astype(str))

    ax.set_xlabel("Model bucket")
    ax.set_ylabel("Original bucket")
    ax.set_title("Row-normalized crosstab (% of each original bucket)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of original bucket")

    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t * 100:.0f}%" for t in ticks])

    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            val = pct.iat[i, j]
            label = "" if pd.isna(val) else f"{val * 100:.1f}%"
            ax.text(j, i, label, ha="center", va="center", color="black")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    return fig


def plot_top_features(imp: pd.DataFrame, top_n=25, which="mean_abs_coef", savepath=None):
    """
    Plot the top features ranked by a chosen importance metric.

    Common choices for `which`:
    - mean_abs_coef
    - sign_consistency

    Parameters
    ----------
    imp : pd.DataFrame
        Feature importance summary indexed by feature name.
    top_n : int, default=25
        Number of top features to display.
    which : str, default="mean_abs_coef"
        Column in `imp` used for ranking.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure()

    s = imp[which].sort_values(ascending=False).head(top_n)

    plt.barh(np.arange(len(s)), s.values)
    plt.yticks(np.arange(len(s)), [pretty_feature_name(i) for i in s.index])
    plt.gca().invert_yaxis()
    plt.xlabel(which)
    plt.title(f"Top {top_n} features by {which}")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_top_pos_neg_features(imp: pd.DataFrame, top_n=20, savepath_pos=None, savepath_neg=None):
    """
    Plot the most positive and most negative mean coefficients separately.

    Positive coefficients push predictions toward Win=1.
    Negative coefficients push predictions toward Win=0.

    Parameters
    ----------
    imp : pd.DataFrame
        Feature importance summary containing mean_coef.
    top_n : int, default=20
        Number of positive and negative features to show.
    savepath_pos : str or Path, optional
        File path to save the positive-coefficient chart.
    savepath_neg : str or Path, optional
        File path to save the negative-coefficient chart.

    Returns
    -------
    tuple
        (fig1, fig2) for positive and negative plots.
    """
    fig1 = plt.figure()
    pos = imp["mean_coef"].sort_values(ascending=False).head(top_n)

    plt.barh(np.arange(len(pos)), pos.values)
    plt.yticks(np.arange(len(pos)), [pretty_feature_name(i) for i in pos.index])
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} positive mean coefficients (toward Win=1)")
    plt.tight_layout()

    if savepath_pos:
        plt.savefig(savepath_pos, dpi=200)

    fig2 = plt.figure()
    neg = imp["mean_coef"].sort_values(ascending=True).head(top_n)

    plt.barh(np.arange(len(neg)), neg.values)
    plt.yticks(np.arange(len(neg)), [pretty_feature_name(i) for i in neg.index])
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} negative mean coefficients (toward Win=0)")
    plt.tight_layout()

    if savepath_neg:
        plt.savefig(savepath_neg, dpi=200)

    return fig1, fig2


def plot_metrics_table(y_true: np.ndarray, proba: np.ndarray, threshold: float, outdir: Path) -> None:
    """
    Create and save a table-style summary image of pooled classification metrics.

    Metrics include:
    - ROC-AUC
    - Average Precision
    - Threshold
    - TPR / FPR / TNR / FNR
    - TP / FP / TN / FN counts

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    proba : np.ndarray
        Predicted probabilities for class 1.
    threshold : float
        Threshold used to convert probabilities to binary predictions.
    outdir : Path or str
        Directory where metrics_table.png will be saved.

    Returns
    -------
    None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)

    rows = [
        ["ROC-AUC", f"{auc:.3f}"],
        ["Avg Precision (PR-AUC)", f"{ap:.3f}"],
        ["Threshold", f"{threshold:.3f}"],
        ["TPR (Recall)", f"{tpr:.3f}"],
        ["FPR", f"{fpr:.3f}"],
        ["TNR (Specificity)", f"{tnr:.3f}"],
        ["FNR", f"{fnr:.3f}"],
        ["TP", f"{tp:d}"],
        ["FP", f"{fp:d}"],
        ["TN", f"{tn:d}"],
        ["FN", f"{fn:d}"],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axis("off")
    ax.set_title("Pooled Metrics Summary", pad=12)

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.35)

    for j in range(2):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.15)

    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(2):
                tbl[i, j].set_alpha(0.85)

    fig.tight_layout()
    fig.savefig(outdir / "metrics_table.png", dpi=200)
    plt.close(fig)


def bucket_summary_table(
    results: pd.DataFrame,
    bucket_col: str,
    win_col: str = "Win",
    proba_col: str = "proba_win",
) -> pd.DataFrame:
    """
    Build a per-bucket summary table for calibration-style analysis.

    For each bucket, computes:
    - n_rows
    - win_rate
    - avg_proba
    - win_pct
    - avg_proba_pct

    Parameters
    ----------
    results : pd.DataFrame
        Candidate-level results.
    bucket_col : str
        Name of the bucket column to group by.
    win_col : str, default="Win"
        Name of the observed binary outcome column.
    proba_col : str, default="proba_win"
        Name of the predicted probability column.

    Returns
    -------
    pd.DataFrame
        Per-bucket summary table.
    """
    df = results[[bucket_col, win_col, proba_col]].copy()
    df[win_col] = pd.to_numeric(df[win_col], errors="coerce")
    df[proba_col] = pd.to_numeric(df[proba_col], errors="coerce")

    out = (
        df.groupby(bucket_col, dropna=False, observed=False)
        .agg(
            n_rows=(win_col, "count"),
            win_rate=(win_col, "mean"),
            avg_proba=(proba_col, "mean"),
        )
        .reset_index()
    )

    out["win_pct"] = 100.0 * out["win_rate"]
    out["avg_proba_pct"] = 100.0 * out["avg_proba"]
    return out.sort_values(bucket_col)


def plot_bucket_calibration(results: pd.DataFrame, bucket_col: str, savepath=None):
    """
    Plot observed win rate vs average predicted probability by bucket.

    This is a simple grouped-bar calibration view at the bucket level.

    Parameters
    ----------
    results : pd.DataFrame
        Candidate-level results containing bucket, outcome, and probability columns.
    bucket_col : str
        Bucket column to summarize.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure()
    tbl = bucket_summary_table(results, bucket_col)

    x = np.arange(len(tbl))
    w = 0.4

    plt.bar(x - w / 2, tbl["win_pct"].values, width=w, label="Observed win %")
    plt.bar(x + w / 2, tbl["avg_proba_pct"].values, width=w, label="Avg predicted %")
    plt.xticks(x, tbl[bucket_col].astype(str).values, rotation=30, ha="right")
    plt.ylabel("Percent")
    plt.title(f"Bucket calibration: {bucket_col}")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig


def plot_diverging_top_features(imp: pd.DataFrame, top_each=10, savepath=None):
    """
    Plot the strongest negative and positive coefficients on one diverging chart.

    Parameters
    ----------
    imp : pd.DataFrame
        Feature importance summary containing mean_coef.
    top_each : int, default=10
        Number of strongest negative and strongest positive features to include.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    df = imp.copy()
    df = df[df["mean_coef"].notna()]

    pos = df.sort_values("mean_coef", ascending=False).head(top_each)
    neg = df.sort_values("mean_coef", ascending=True).head(top_each)
    d = pd.concat([neg, pos], axis=0)
    d = d.reindex(d["mean_coef"].abs().sort_values().index)

    labels = [pretty_feature_name(i) for i in d.index]
    vals = d["mean_coef"].values

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(d))
    ax.barh(y, vals)
    ax.axvline(0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean coefficient (log-odds)")
    ax.set_title(f"Top drivers (mean coefficients): {top_each} negative + {top_each} positive")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


def plot_grouped_small_multiples(imp: pd.DataFrame, per_group=10, savepath=None):
    """
    Plot top features by domain using small multiples.

    Each subplot shows the top features within one domain, using a common x-axis
    scale to make coefficient magnitudes more comparable across domains.

    Parameters
    ----------
    imp : pd.DataFrame
        Feature importance summary containing mean_coef and mean_abs_coef.
    per_group : int, default=10
        Number of features to show per domain.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    df = imp.copy().dropna(subset=["mean_coef"])
    df["domain"] = [feature_domain(i) for i in df.index]
    df = df.sort_values("mean_abs_coef", ascending=False)

    domains = [
        d for d in [
            "Race structure",
            "Office",
            "Geography",
            "Election timing",
            "Outreach",
            "Election Type",
        ]
        if d in df["domain"].unique()
    ]

    plotted = []
    for dom in domains:
        sub = df[df["domain"] == dom].head(per_group).copy()
        plotted.append(sub)

    plotted_df = pd.concat(plotted) if plotted else df.head(per_group)

    m = float(plotted_df["mean_coef"].abs().max())
    m = max(m, 1e-6)
    pad = 0.10
    xlim = (-m * (1 + pad), m * (1 + pad))

    n = len(domains)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, dom in zip(axes, domains):
        sub = df[df["domain"] == dom].head(per_group).copy()
        sub = sub.reindex(sub["mean_abs_coef"].sort_values().index)

        y = np.arange(len(sub))
        ax.barh(y, sub["mean_coef"].values)
        ax.axvline(0, linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels([pretty_feature_name(i) for i in sub.index])
        ax.set_title(dom)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Mean coefficient")

    fig.suptitle(
        "Top effects by feature domain (common x-axis scale)",
        y=1.01,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


def plot_structural_vs_state_effects(imp: pd.DataFrame, top_struct=12, top_states=8, savepath=None):
    """
    Plot non-state structural drivers separately from state effects.

    This is useful because state dummy coefficients can be visually dominant
    and easy to over-interpret. Separating them helps tell a cleaner story.

    Parameters
    ----------
    imp : pd.DataFrame
        Feature importance summary containing mean_coef and mean_abs_coef.
    top_struct : int, default=12
        Number of non-state structural features to show.
    top_states : int, default=8
        Number of state features to show.
    savepath : str or Path, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    df = imp.copy().dropna(subset=["mean_coef"])

    states = df[df.index.to_series().astype(str).str.startswith("state_usps_")].copy()
    non_states = df[~df.index.to_series().astype(str).str.startswith("state_usps_")].copy()

    struct = non_states.sort_values("mean_abs_coef", ascending=False).head(top_struct).copy()
    struct = struct.reindex(struct["mean_abs_coef"].sort_values().index)

    st = states.sort_values("mean_abs_coef", ascending=False).head(top_states).copy()
    st = st.reindex(st["mean_abs_coef"].sort_values().index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    ax = axes[0]
    y = np.arange(len(struct))
    ax.barh(y, struct["mean_coef"].values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(i) for i in struct.index])
    ax.set_title("Structural drivers (non-state)")
    ax.set_xlabel("Mean coefficient")

    ax = axes[1]
    y = np.arange(len(st))
    ax.barh(y, st["mean_coef"].values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(i) for i in st.index])
    ax.set_title("State effects (top by |coef|)")
    ax.set_xlabel("Mean coefficient")

    fig.suptitle(
        "Separate structural effects from geography (reduces over-interpretation of states)",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


# =========================================================
# Main entry point
# =========================================================

def make_all_plots(
        fold_metrics_df: pd.DataFrame,
        oof_df: pd.DataFrame,
        threshold: float,
        results: pd.DataFrame,
        imp: pd.DataFrame,
        outdir="viz_outputs",
    ):
    """
    Generate and save the full standard set of model evaluation plots.

    This is the main orchestrator for the module. It:
    1. Reads pooled OOF labels and predictions
    2. Creates prediction diagnostics if probabilities are available
    3. Creates fold ROC-AUC summary if fold metrics are available
    4. Creates viability comparison plots if bucket columns are present
    5. Creates feature importance plots if importance data is provided

    Parameters
    ----------
    fold_metrics_df : pd.DataFrame
        Fold-level metrics. Expected to include roc_auc if available.
    oof_df : pd.DataFrame
        Out-of-fold predictions with expected columns such as:
        - y_true
        - pred_score (optional)
        - pred_proba (optional)
    threshold : float
        Classification threshold for binary plots.
    results : pd.DataFrame
        Candidate-level results used for viability and calibration plots.
    imp : pd.DataFrame
        Aggregated feature importance summary.
    outdir : str or Path, default="viz_outputs"
        Directory where all plots will be saved.

    Returns
    -------
    None
        All plots are saved to disk. Figures are closed at the end.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y_all = np.asarray(oof_df["y_true"])

    pred_score = None
    if "pred_score" in oof_df.columns:
        pred_score = np.asarray(oof_df["pred_score"])

    pred_proba = None
    if "pred_proba" in oof_df.columns:
        pred_proba = np.asarray(oof_df["pred_proba"])

    # Probability-specific plots require calibrated/predict_proba-style outputs.
    score_for_probability_plots = pred_proba

    # AUC can be computed from either a raw decision score or probability.
    score_for_auc = pred_score if pred_score is not None else pred_proba

    # ----------------------------
    # Probability-based plots
    # ----------------------------
    if score_for_probability_plots is not None:
        plot_score_hist_by_class(
            y_all,
            score_for_probability_plots,
            threshold,
            savepath=outdir / "score_hist_by_class.png",
        )

        plot_confusion_matrix(
            y_all,
            score_for_probability_plots,
            threshold,
            normalize=False,
            savepath=outdir / "confusion_matrix.png",
        )

        plot_confusion_matrix(
            y_all,
            score_for_probability_plots,
            threshold,
            normalize=True,
            savepath=outdir / "confusion_matrix_normalized.png",
        )

        plot_threshold_sweep(
            y_all,
            score_for_probability_plots,
            savepath=outdir / "threshold_sweep.png",
        )

        plot_metrics_table(
            y_all,
            score_for_probability_plots,
            threshold,
            outdir,
        )

    # ----------------------------
    # Fold AUC summary plot
    # ----------------------------
    if fold_metrics_df is not None and "roc_auc" in fold_metrics_df.columns:
        auc_vals = pd.to_numeric(fold_metrics_df["roc_auc"], errors="coerce").dropna()

        if len(auc_vals) > 0:
            fig = plt.figure()
            plt.bar(range(1, len(auc_vals) + 1), auc_vals.values)
            plt.xlabel("Fold")
            plt.ylabel("ROC-AUC")
            plt.title("Fold ROC-AUC")
            plt.xticks(range(1, len(auc_vals) + 1))
            plt.tight_layout()
            plt.savefig(outdir / "fold_auc_bar.png", dpi=200)

    # ----------------------------
    # Viability plots
    # ----------------------------
    required_cols = {"viability_bucket_orig", "viability_bucket_model", "bucket_distance"}
    if required_cols.issubset(results.columns):
        plot_viability_bucket_distributions(
            results,
            savepath=outdir / "bucket_distribution.png",
        )
        plot_bucket_crosstab_heatmap(
            results,
            savepath=outdir / "bucket_crosstab_heatmap.png",
        )
        plot_bucket_distance(
            results,
            savepath=outdir / "bucket_distance.png",
        )

    if {"Win", "proba_win", "viability_bucket_model"}.issubset(results.columns):
        plot_bucket_calibration(
            results,
            "viability_bucket_model",
            savepath=outdir / "bucket_calibration_model.png",
        )

    # ----------------------------
    # Feature importance plots
    # ----------------------------
    if imp is not None and len(imp) > 0:
        if "mean_abs_coef" in imp.columns:
            plot_top_features(
                imp,
                top_n=25,
                which="mean_abs_coef",
                savepath=outdir / "top_features_mean_abs_coef.png",
            )

        if "sign_consistency" in imp.columns:
            plot_top_features(
                imp,
                top_n=25,
                which="sign_consistency",
                savepath=outdir / "top_features_sign_consistency.png",
            )

        if "mean_coef" in imp.columns:
            plot_top_pos_neg_features(
                imp,
                top_n=20,
                savepath_pos=outdir / "top_positive_mean_coef.png",
                savepath_neg=outdir / "top_negative_mean_coef.png",
            )
            plot_diverging_top_features(
                imp,
                top_each=10,
                savepath=outdir / "coef_diverging_top10_each.png",
            )
            plot_grouped_small_multiples(
                imp,
                per_group=10,
                savepath=outdir / "coef_small_multiples_by_domain.png",
            )
            plot_structural_vs_state_effects(
                imp,
                top_struct=12,
                top_states=8,
                savepath=outdir / "coef_structural_vs_states.png",
            )

    plt.close("all")