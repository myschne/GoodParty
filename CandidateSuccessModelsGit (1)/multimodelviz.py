"""
Multimodel visualization utilities for Candidate Success modeling.

This module loads saved model artifacts from Unity Catalog tables and produces
comparison visualizations across several model types. It is designed to work
after training has already completed, so you can regenerate plots without
retraining models.

What this script currently does
-------------------------------
1. Loads out-of-fold prediction tables, fold metric tables, and feature catalogs
   for each model from Unity Catalog.
2. Builds a current-threshold predictive metrics summary table.
3. Produces one readable confusion matrix per model.
4. Produces two consolidated threshold sweep figures:
   - confusion-matrix counts across thresholds
   - precision / accuracy / recall across thresholds
5. Produces predicted-probability histograms by actual class.
6. Produces viability / bucket comparison plots.
7. Produces coefficient-style feature plots for linear models only
   (logistic regression and elastic net), grouped by structural domains.

Design notes
------------
- Threshold-based ROC-AUC plots are intentionally omitted because ROC-AUC does
  not vary with threshold.
- Feature plots are intentionally limited to linear models because signed
  coefficients are directly interpretable there, while tree-model importances
  are not directly comparable on the same signed scale.
- The script is conservative about missing tables: it prints warnings and
  continues when possible.

Expected Unity Catalog tables
-----------------------------
For each model name, this script expects:
- {catalog}.{schema}.{model_name}_oof_predictions
- {catalog}.{schema}.{model_name}_fold_metrics
- {catalog}.{schema}.{model_name}_feature_catalog

Optionally, if include_results=True:
- {catalog}.{schema}.{model_name}_results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import THRESHOLD, UC_CATALOG, UC_SCHEMA


# ============================================================================
# Global configuration
# ============================================================================

MODEL_NAMES = [
    "logistic_regression",
    "elastic_net_logistic",
    "random_forest",
    "xgboost",
]

# Only these models get coefficient-style feature plots
LINEAR_MODELS = [
    "logistic_regression",
    "elastic_net_logistic",
]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic",
    "elastic_net_logistic": "Elastic Net",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

VIAB_LABELS = [
    "No Chance",
    "Unlikely to Win",
    "Has a Chance",
    "Likely to Win",
    "Frontrunner",
]

# Threshold grid used for sweep plots
DEFAULT_SWEEP_THRESHOLDS = np.linspace(0.05, 0.95, 19)

# Softer, more presentation-friendly colormap for confusion matrices
CONFUSION_MATRIX_CMAP = "Blues"


# ============================================================================
# Basic helpers
# ============================================================================

def pretty_model_name(model_name: str) -> str:
    """Return a cleaner display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _safe_copy(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a defensive copy, or an empty DataFrame if input is None."""
    return pd.DataFrame() if df is None else df.copy()


def _ensure_dir(outdir: str | Path) -> None:
    """Create output directory if it does not already exist."""
    Path(outdir).mkdir(parents=True, exist_ok=True)


def _find_first_existing_col(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """
    Return the first matching column name from a candidate list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    candidates : list[str]
        Ordered list of candidate column names.
    required : bool, default=True
        If True, raise KeyError when no candidate is found.

    Returns
    -------
    str | None
        First matching column name, or None if not found and required=False.
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None


def _zscore(s: pd.Series) -> pd.Series:
    """
    Z-score normalize a numeric series.

    Returns zeros if variance is zero or values are invalid.
    """
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def _bucket_to_num(series: pd.Series) -> pd.Series:
    """
    Convert ordered viability labels to 1..5 numeric bucket values.
    """
    cat = pd.Categorical(series, categories=VIAB_LABELS, ordered=True)
    return pd.Series(cat.codes + 1, index=series.index)


# ============================================================================
# Unity Catalog table naming / loading
# ============================================================================

def get_oof_table(model_name: str) -> str:
    """Return the fully qualified UC table name for OOF predictions."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_oof_predictions"


def get_fold_metrics_table(model_name: str) -> str:
    """Return the fully qualified UC table name for fold metrics."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_fold_metrics"


def get_feature_catalog_table(model_name: str) -> str:
    """Return the fully qualified UC table name for feature catalog."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_feature_catalog"


def get_results_table(model_name: str) -> str:
    """Return the fully qualified UC table name for optional results table."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_results"


def safe_read_table(spark, table_name: str) -> pd.DataFrame:
    """
    Read a Spark table into pandas, returning an empty DataFrame on failure.

    This keeps plot generation robust when some saved artifacts are missing.
    """
    try:
        return spark.table(table_name).toPandas()
    except Exception as exc:
        print(f"Could not read {table_name}: {exc}")
        return pd.DataFrame()


def load_model_outputs_from_uc(
    spark,
    model_names: list[str] = MODEL_NAMES,
    include_results: bool = False,
) -> dict:
    """
    Load saved model artifacts from Unity Catalog into a common structure.

    Returns
    -------
    dict
        {
            model_name: {
                "oof_df": pd.DataFrame,
                "fold_metrics_df": pd.DataFrame,
                "feature_catalog_df": pd.DataFrame,
                "imp": pd.DataFrame,
                "results": pd.DataFrame,  # optional
            }
        }
    """
    model_outputs = {}

    for model_name in model_names:
        oof_df = safe_read_table(spark, get_oof_table(model_name))
        fold_metrics_df = safe_read_table(spark, get_fold_metrics_table(model_name))
        feature_catalog_df = safe_read_table(spark, get_feature_catalog_table(model_name))

        payload = {
            "oof_df": oof_df,
            "fold_metrics_df": fold_metrics_df,
            "feature_catalog_df": feature_catalog_df,
            # Fallback: use feature catalog as the importance source if no
            # separate importance table exists.
            "imp": feature_catalog_df.copy() if not feature_catalog_df.empty else pd.DataFrame(),
        }

        if include_results:
            payload["results"] = safe_read_table(spark, get_results_table(model_name))

        model_outputs[model_name] = payload

    return model_outputs


def get_thresholds(
    model_names: list[str] = MODEL_NAMES,
    default_threshold: float = THRESHOLD,
) -> dict[str, float]:
    """
    Return a threshold dictionary for each model.

    Current behavior applies the same threshold to every model.
    """
    return {model_name: default_threshold for model_name in model_names}


def validate_model_outputs(model_outputs: dict) -> None:
    """
    Run lightweight validation checks before plotting.

    Currently checks:
    - oof_df exists and has y_true / pred_proba when present
    - warns if feature catalogs are missing
    """
    required_oof_cols = {"y_true", "pred_proba"}

    for model_name, payload in model_outputs.items():
        oof_df = payload.get("oof_df", pd.DataFrame())
        if oof_df.empty:
            print(f"[WARN] {model_name}: oof_df is empty")
            continue

        missing = required_oof_cols - set(oof_df.columns)
        if missing:
            raise ValueError(
                f"{model_name} oof_df is missing required columns: {sorted(missing)}"
            )

        feature_catalog_df = payload.get("feature_catalog_df", pd.DataFrame())
        if feature_catalog_df.empty:
            print(f"[WARN] {model_name}: feature_catalog_df is empty")


# ============================================================================
# Feature naming / grouping helpers
# ============================================================================

def pretty_feature_name(feature: str) -> str:
    """
    Convert raw engineered feature names into cleaner presentation labels.
    """
    feature = str(feature)

    replacements = {
        "state_usps_": "State: ",
        "region_": "Region: ",
        "office_level_clean_": "Office: ",
        "election_dow_": "Election day: ",
        "is_midterm": "Midterm election",
        "is_presidential": "Presidential election",
        "is_normal_election": "Normal November election",
        "number_of_opponents_num": "# Opponents",
        "number_avail_seats": "# Seats",
        "recency_weighted_days": "Recency weighted outreach",
        "days_between_outreach_and_election": "Days between outreach and election",
        "recency_election_interaction": "Outreach × recency",
    }

    out = feature
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _domain_from_feature_name(feature: str) -> str:
    """
    Map a feature name to a broader feature domain.

    This is a heuristic mapping based on naming conventions in the pipeline.
    """
    f = str(feature).lower()

    if any(x in f for x in ["opponent", "seat", "incumb", "competit"]):
        return "Race structure"
    if any(x in f for x in ["office", "federal", "state_", "local", "partisan"]):
        return "Office"
    if any(x in f for x in ["state_usps", "region", "west", "south", "midwest", "northeast", "territory"]):
        return "Geography"
    if any(x in f for x in ["election_", "midterm", "presidential", "normal_election", "_dow_"]):
        return "Election timing"
    if any(x in f for x in ["outreach", "message", "script", "persona", "recency", "days_between"]):
        return "Outreach"
    if any(x in f for x in ["general", "runoff", "special", "primary"]):
        return "Election Type"
    if any(x in f for x in ["sentiment", "angry", "positive", "negative"]):
        return "Text / sentiment"
    return "Other"


def feature_domain(feature: str) -> str:
    """Public wrapper for feature-domain mapping."""
    return _domain_from_feature_name(feature)


# ============================================================================
# Viability / bucket helpers
# ============================================================================

def _add_bucket_columns(df: pd.DataFrame, proba_col: str = "pred_proba") -> pd.DataFrame:
    """
    Add model-derived viability score and bucket columns.

    Adds:
    - viability_score_model: probability scaled to 0..5
    - viability_bucket_model: ordered text label
    - viability_bucket_num_model: ordered integer bucket 1..5
    """
    df = df.copy()

    if proba_col not in df.columns:
        raise KeyError(f"{proba_col} not found in DataFrame")

    df["viability_score_model"] = 5.0 * pd.to_numeric(df[proba_col], errors="coerce")
    df["viability_bucket_model"] = pd.cut(
        df["viability_score_model"],
        bins=[0, 1, 2, 3, 4, 5],
        labels=VIAB_LABELS,
        include_lowest=True,
        right=True,
    )
    df["viability_bucket_num_model"] = (
        pd.Categorical(df["viability_bucket_model"], categories=VIAB_LABELS, ordered=True).codes + 1
    )
    return df


def _find_original_bucket_col(df: pd.DataFrame) -> str | None:
    """
    Find the original viability bucket column if present.
    """
    return _find_first_existing_col(
        df,
        [
            "viability_bucket_original",
            "viability_bucket_orig",
            "viability_bucket",
            "original_viability_bucket",
            "viability_label",
        ],
        required=False,
    )


# ============================================================================
# Predictive metrics
# ============================================================================

def build_predictive_metrics_table(
    model_outputs: dict,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Build a one-row-per-model metrics table at each model's current threshold.

    Includes:
    - TP, TN, FP, FN
    - accuracy, precision, recall, F1
    - ROC-AUC
    """
    rows = []

    for model_name, payload in model_outputs.items():
        oof_df = _safe_copy(payload.get("oof_df"))
        if oof_df.empty:
            continue

        y_true = oof_df["y_true"].astype(int).to_numpy()
        pred_proba = pd.to_numeric(oof_df["pred_proba"], errors="coerce").to_numpy()
        threshold = thresholds[model_name]
        pred_label = (pred_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy_score(y_true, pred_label),
                "precision": precision_score(y_true, pred_label, zero_division=0),
                "recall": recall_score(y_true, pred_label, zero_division=0),
                "f1_score": f1_score(y_true, pred_label, zero_division=0),
                "roc_auc": roc_auc_score(y_true, pred_proba),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def plot_predictive_metrics_table(metrics_df: pd.DataFrame, outdir: str | Path) -> None:
    """
    Save a formatted table image of current-threshold predictive metrics.
    """
    _ensure_dir(outdir)

    if metrics_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(2, 0.6 * len(metrics_df) + 1.5)))
    ax.axis("off")

    display_df = metrics_df.copy()
    for col in ["threshold", "accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.title("Predictive Metrics at Current Threshold")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "predictive_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# Confusion matrices
# ============================================================================

def _choose_confusion_text_color(value: float, vmax: float) -> str:
    """
    Choose white or dark text depending on cell intensity for readability.
    """
    if vmax <= 0:
        return "#1f2937"
    return "white" if value > 0.55 * vmax else "#1f2937"


def plot_confusion_matrices_by_model(
    model_outputs: dict,
    thresholds: dict[str, float],
    outdir: str | Path,
) -> None:
    """
    Save one confusion matrix figure per model.

    Styling notes
    -------------
    - Uses a soft blue colormap for easier presentation readability.
    - Uses dynamic text color so counts remain legible.
    - Includes a colorbar for quick visual interpretation.
    """
    _ensure_dir(outdir)

    for model_name, payload in model_outputs.items():
        oof_df = _safe_copy(payload.get("oof_df"))
        if oof_df.empty:
            continue

        y_true = oof_df["y_true"].astype(int).to_numpy()
        pred_proba = pd.to_numeric(oof_df["pred_proba"], errors="coerce").to_numpy()
        threshold = thresholds[model_name]
        pred_label = (pred_proba >= threshold).astype(int)

        cm = confusion_matrix(y_true, pred_label, labels=[0, 1])
        vmax = cm.max()

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm, cmap=CONFUSION_MATRIX_CMAP, vmin=0, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Count", rotation=270, labelpad=14)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted 0", "Predicted 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion Matrix: {pretty_model_name(model_name)}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")

        # Light grid to visually separate cells
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=_choose_confusion_text_color(val, vmax),
                )

        plt.tight_layout()
        plt.savefig(Path(outdir) / f"{model_name}_confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()


# ============================================================================
# Threshold sweeps
# ============================================================================

def build_threshold_sweep_table(
    model_outputs: dict,
    thresholds: np.ndarray = DEFAULT_SWEEP_THRESHOLDS,
) -> pd.DataFrame:
    """
    Build threshold-sweep metrics for each model across a common threshold grid.

    Includes:
    - TP, TN, FP, FN
    - accuracy, precision, recall

    Intentionally excludes ROC-AUC because ROC-AUC does not change with
    threshold and therefore does not belong in a threshold sweep plot.
    """
    rows = []

    for model_name, payload in model_outputs.items():
        oof_df = _safe_copy(payload.get("oof_df"))
        if oof_df.empty:
            continue

        y_true = oof_df["y_true"].astype(int).to_numpy()
        pred_proba = pd.to_numeric(oof_df["pred_proba"], errors="coerce").to_numpy()

        for t in thresholds:
            pred_label = (pred_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()

            rows.append(
                {
                    "model": model_name,
                    "threshold": float(t),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "accuracy": accuracy_score(y_true, pred_label),
                    "precision": precision_score(y_true, pred_label, zero_division=0),
                    "recall": recall_score(y_true, pred_label, zero_division=0),
                }
            )

    return pd.DataFrame(rows)


def plot_threshold_sweep_confusion_counts(
    sweep_df: pd.DataFrame,
    thresholds: dict[str, float],
    outdir: str | Path,
) -> None:
    """
    Save a consolidated 2x2 threshold sweep figure for TP / TN / FP / FN.

    Each panel contains one line per model and a dashed vertical line at that
    model's current operating threshold.
    """
    _ensure_dir(outdir)

    if sweep_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    metric_map = {
        "tp": "True Positives",
        "tn": "True Negatives",
        "fp": "False Positives",
        "fn": "False Negatives",
    }

    for ax, metric in zip(axes, ["tp", "tn", "fp", "fn"]):
        for model_name, g in sweep_df.groupby("model"):
            g = g.sort_values("threshold")

            ax.plot(
                g["threshold"],
                g[metric],
                linewidth=2,
                label=pretty_model_name(model_name),
            )
            ax.axvline(
                thresholds[model_name],
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )

        ax.set_title(metric_map[metric])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Count")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Threshold Sweep: Confusion Matrix Counts by Model",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(Path(outdir) / "threshold_sweep_confusion_counts.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep_classification_metrics(
    sweep_df: pd.DataFrame,
    thresholds: dict[str, float],
    outdir: str | Path,
) -> None:
    """
    Save a consolidated threshold sweep figure for precision, accuracy, recall.

    Each panel contains one line per model and a dashed vertical line at that
    model's current operating threshold.
    """
    _ensure_dir(outdir)

    if sweep_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    metric_map = {
        "precision": "Precision",
        "accuracy": "Accuracy",
        "recall": "Recall",
    }

    for ax, metric in zip(axes, ["precision", "accuracy", "recall"]):
        for model_name, g in sweep_df.groupby("model"):
            g = g.sort_values("threshold")

            ax.plot(
                g["threshold"],
                g[metric],
                linewidth=2,
                label=pretty_model_name(model_name),
            )
            ax.axvline(
                thresholds[model_name],
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )

        ax.set_title(metric_map[metric])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Threshold Sweep: Precision, Accuracy, and Recall by Model",
        y=1.05,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(Path(outdir) / "threshold_sweep_classification_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# Predicted probability plots
# ============================================================================

def plot_predicted_probability_by_class(
    model_outputs: dict,
    thresholds: dict[str, float],
    outdir: str | Path,
) -> None:
    """
    Save predicted-probability histograms split by actual class for each model.
    """
    _ensure_dir(outdir)

    for model_name, payload in model_outputs.items():
        oof_df = _safe_copy(payload.get("oof_df"))
        if oof_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        neg = oof_df.loc[oof_df["y_true"] == 0, "pred_proba"].astype(float)
        pos = oof_df.loc[oof_df["y_true"] == 1, "pred_proba"].astype(float)

        ax.hist(neg, bins=30, alpha=0.5, density=True, label="Actual 0")
        ax.hist(pos, bins=30, alpha=0.5, density=True, label="Actual 1")
        ax.axvline(
            thresholds[model_name],
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {thresholds[model_name]:.2f}",
        )

        ax.set_title(f"Predicted Probability Distribution by Class: {pretty_model_name(model_name)}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        plt.savefig(Path(outdir) / f"{model_name}_predicted_probability_by_class.png", dpi=200, bbox_inches="tight")
        plt.close()


# ============================================================================
# Bucket / viability plots
# ============================================================================

def prepare_results_payload(model_outputs: dict) -> dict:
    """
    Ensure each model has a results-like DataFrame with derived model buckets.

    If payload['results'] exists, use it.
    Otherwise derive from oof_df by adding model viability columns.
    """
    prepared = {}

    for model_name, payload in model_outputs.items():
        results = payload.get("results")

        if results is None or len(results) == 0:
            oof_df = _safe_copy(payload.get("oof_df"))
            if oof_df.empty:
                continue
            results = _add_bucket_columns(oof_df, proba_col="pred_proba")
        else:
            results = _safe_copy(results)
            if "viability_bucket_model" not in results.columns:
                results = _add_bucket_columns(results, proba_col="pred_proba")

        prepared[model_name] = results

    return prepared


def plot_bucket_calibration(prepared_results: dict, outdir: str | Path) -> None:
    """
    Plot observed win rate by model-derived viability bucket.
    """
    _ensure_dir(outdir)

    rows = []
    overall_rows = []

    for model_name, results in prepared_results.items():
        tmp = results.copy()
        tmp["y_true"] = tmp["y_true"].astype(int)

        grouped = (
            tmp.groupby("viability_bucket_model", dropna=False)
            .agg(
                observed_win_pct=("y_true", "mean"),
                avg_predicted_proba=("pred_proba", "mean"),
                n=("y_true", "size"),
            )
            .reset_index()
        )

        grouped["model"] = model_name
        grouped["bucket_num"] = _bucket_to_num(grouped["viability_bucket_model"])
        rows.append(grouped)

        overall_rows.append(
            {
                "model": model_name,
                "overall_actual_win_pct": tmp["y_true"].mean(),
            }
        )

    if not rows:
        return

    cal_df = pd.concat(rows, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.18
    model_list = list(cal_df["model"].unique())
    x = np.arange(len(VIAB_LABELS))

    for i, model_name in enumerate(model_list):
        g = cal_df[cal_df["model"] == model_name].set_index("viability_bucket_model").reindex(VIAB_LABELS)
        ax.bar(x + i * width, g["observed_win_pct"].fillna(0), width=width, label=pretty_model_name(model_name))

    overall_mean = overall_df["overall_actual_win_pct"].mean()
    ax.axhline(
        overall_mean,
        linestyle="--",
        linewidth=2,
        label=f"Avg actual win % across models = {overall_mean:.2%}",
    )

    ax.set_xticks(x + width * (len(model_list) - 1) / 2)
    ax.set_xticklabels(VIAB_LABELS, rotation=20, ha="right")
    ax.set_ylabel("Observed win %")
    ax.set_title("Bucket Calibration: Observed Win % by Model Bucket")
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(outdir) / "bucket_calibration_observed_win_pct.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_bucket_distance_distribution(prepared_results: dict, outdir: str | Path) -> None:
    """
    Plot distribution of absolute distance between original and model buckets.
    """
    _ensure_dir(outdir)

    rows = []

    for model_name, results in prepared_results.items():
        orig_bucket_col = _find_original_bucket_col(results)
        if orig_bucket_col is None:
            continue

        tmp = results.copy()
        tmp["orig_bucket_num"] = _bucket_to_num(tmp[orig_bucket_col])
        tmp["model_bucket_num"] = _bucket_to_num(tmp["viability_bucket_model"])
        tmp["bucket_distance"] = (tmp["orig_bucket_num"] - tmp["model_bucket_num"]).abs()
        tmp["model"] = model_name

        rows.append(tmp[["model", "bucket_distance"]])

    if not rows:
        print("No original bucket column found; skipping bucket distance distribution.")
        return

    dist_df = pd.concat(rows, ignore_index=True)
    summary = dist_df.groupby(["model", "bucket_distance"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, g in summary.groupby("model"):
        ax.plot(
            g["bucket_distance"],
            g["count"],
            marker="o",
            linewidth=2,
            label=pretty_model_name(model_name),
        )

    ax.set_title("Bucket Distance Distribution by Model")
    ax.set_xlabel("Absolute bucket distance")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(outdir) / "bucket_distance_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_bucket_distribution(prepared_results: dict, outdir: str | Path) -> None:
    """
    Plot model-derived bucket distribution and original bucket distribution.
    """
    _ensure_dir(outdir)

    model_rows = []
    orig_rows = []

    for model_name, results in prepared_results.items():
        model_counts = (
            results["viability_bucket_model"]
            .value_counts(dropna=False)
            .reindex(VIAB_LABELS, fill_value=0)
            .reset_index()
        )
        model_counts.columns = ["bucket", "count"]
        model_counts["model"] = model_name
        model_rows.append(model_counts)

        orig_bucket_col = _find_original_bucket_col(results)
        if orig_bucket_col is not None:
            orig_counts = (
                results[orig_bucket_col]
                .value_counts(dropna=False)
                .reindex(VIAB_LABELS, fill_value=0)
                .reset_index()
            )
            orig_counts.columns = ["bucket", "count"]
            orig_counts["model"] = "original"
            orig_rows.append(orig_counts)

    if model_rows:
        all_model_counts = pd.concat(model_rows, ignore_index=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.18
        model_list = list(all_model_counts["model"].unique())
        x = np.arange(len(VIAB_LABELS))

        for i, model_name in enumerate(model_list):
            g = all_model_counts[all_model_counts["model"] == model_name].set_index("bucket").reindex(VIAB_LABELS)
            ax.bar(x + i * width, g["count"], width=width, label=pretty_model_name(model_name))

        ax.set_xticks(x + width * (len(model_list) - 1) / 2)
        ax.set_xticklabels(VIAB_LABELS, rotation=20, ha="right")
        ax.set_title("Model-Derived Bucket Distribution")
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        plt.savefig(Path(outdir) / "bucket_distribution_models.png", dpi=200, bbox_inches="tight")
        plt.close()

    if orig_rows:
        original_counts = pd.concat(orig_rows, ignore_index=True)
        original_counts = (
            original_counts.groupby("bucket", as_index=False)["count"]
            .mean()
            .reindex(columns=["bucket", "count"])
        )
        original_counts = original_counts.set_index("bucket").reindex(VIAB_LABELS).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(original_counts["bucket"], original_counts["count"])
        ax.set_title("Original Bucket Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.savefig(Path(outdir) / "bucket_distribution_original.png", dpi=200, bbox_inches="tight")
        plt.close()


# ============================================================================
# Linear-model feature plots
# ============================================================================

def plot_grouped_small_multiples(
    imp: pd.DataFrame,
    per_group: int = 10,
    savepath: str | Path | None = None,
):
    """
    Plot top features by domain using stacked small multiples.

    Each subplot uses a common x-axis scale so coefficient magnitudes can be
    compared across domains more fairly.
    """
    df = imp.copy().dropna(subset=["mean_coef"])
    df["domain"] = [feature_domain(i) for i in df.index]
    df = df.sort_values("mean_abs_coef", ascending=False)

    domains = [
        d
        for d in [
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
    xlim = (-m * 1.10, m * 1.10)

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
        "Top Effects by Feature Domain (Common X-Axis Scale)",
        y=1.01,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


def plot_structural_vs_state_effects(
    imp: pd.DataFrame,
    top_struct: int = 12,
    top_states: int = 8,
    savepath: str | Path | None = None,
):
    """
    Plot non-state structural drivers separately from state effects.

    This helps prevent state dummy variables from visually dominating the story.
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
    ax.set_title("Structural Drivers (Non-State)")
    ax.set_xlabel("Mean coefficient")

    ax = axes[1]
    y = np.arange(len(st))
    ax.barh(y, st["mean_coef"].values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(i) for i in st.index])
    ax.set_title("State Effects (Top by |coef|)")
    ax.set_xlabel("Mean coefficient")

    fig.suptitle(
        "Separate Structural Effects from Geography",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


def plot_linear_feature_groups(
    model_outputs: dict,
    outdir: str | Path,
    per_group: int = 8,
) -> None:
    """
    Generate grouped coefficient plots for linear models only.

    For each linear model:
    - grouped small multiples by feature domain
    - structural vs state effects split
    """
    _ensure_dir(outdir)

    for model_name in LINEAR_MODELS:
        payload = model_outputs.get(model_name, {})
        imp = _safe_copy(payload.get("imp"))

        if imp.empty:
            imp = _safe_copy(payload.get("feature_catalog_df"))
        if imp.empty:
            continue

        imp = imp.copy()

        if "feature" not in imp.columns:
            candidate = _find_first_existing_col(
                imp,
                ["feature_name", "feature", "term"],
                required=False,
            )
            if candidate is not None:
                imp["feature"] = imp[candidate]

        coef_col = _find_first_existing_col(
            imp,
            ["mean_coef", "coefficient", "coef", "mean_importance", "importance"],
            required=False,
        )
        if coef_col is None:
            continue

        imp["mean_coef"] = pd.to_numeric(imp[coef_col], errors="coerce")
        imp["mean_abs_coef"] = imp["mean_coef"].abs()
        imp = imp.dropna(subset=["feature", "mean_coef"]).set_index("feature")

        plot_grouped_small_multiples(
            imp=imp,
            per_group=per_group,
            savepath=Path(outdir) / f"{model_name}_top_effects_by_feature_domain.png",
        )

        plot_structural_vs_state_effects(
            imp=imp,
            top_struct=12,
            top_states=8,
            savepath=Path(outdir) / f"{model_name}_structural_vs_state_effects.png",
        )


# ============================================================================
# Orchestration
# ============================================================================

def make_all_multimodel_plots(
    model_outputs: dict,
    thresholds: dict[str, float],
    outdir: str | Path,
) -> dict:
    """
    Generate the full multimodel visualization suite.

    Active outputs
    --------------
    - predictive_metrics_table.png
    - one confusion_matrix.png per model
    - threshold_sweep_confusion_counts.png
    - threshold_sweep_classification_metrics.png
    - predicted_probability_by_class.png per model
    - viability / bucket plots
    - linear-model domain coefficient plots
    """
    _ensure_dir(outdir)

    metrics_df = build_predictive_metrics_table(model_outputs, thresholds)
    plot_predictive_metrics_table(metrics_df, outdir)

    plot_confusion_matrices_by_model(model_outputs, thresholds, outdir)

    sweep_df = build_threshold_sweep_table(model_outputs)
    plot_threshold_sweep_confusion_counts(sweep_df, thresholds, outdir)
    plot_threshold_sweep_classification_metrics(sweep_df, thresholds, outdir)

    plot_predicted_probability_by_class(model_outputs, thresholds, outdir)

    prepared_results = prepare_results_payload(model_outputs)
    plot_bucket_calibration(prepared_results, outdir)
    plot_bucket_distance_distribution(prepared_results, outdir)
    plot_bucket_distribution(prepared_results, outdir)

    plot_linear_feature_groups(model_outputs, outdir, per_group=8)

    return {
        "predictive_metrics_df": metrics_df,
        "threshold_sweep_df": sweep_df,
    }


def main(
    spark,
    model_names: list[str] = MODEL_NAMES,
    include_results: bool = False,
    outdir: str = "/Workspace/Users/myschne@umich.edu/CandidateSuccessModels/multimodel_viz_outputs",
) -> dict:
    """
    Load saved model artifacts from Unity Catalog and generate plots.

    Parameters
    ----------
    spark : SparkSession
        Active Databricks Spark session.
    model_names : list[str], default=MODEL_NAMES
        Models to include in the comparison suite.
    include_results : bool, default=False
        Whether to also load saved *_results tables.
    outdir : str
        Output directory for image files.

    Returns
    -------
    dict
        {
            "model_outputs": ...,
            "thresholds": ...,
            "viz_outputs": ...,
            "outdir": ...,
        }
    """
    print("Loading model outputs from Unity Catalog tables...")
    model_outputs = load_model_outputs_from_uc(
        spark=spark,
        model_names=model_names,
        include_results=include_results,
    )

    print("Validating loaded tables...")
    validate_model_outputs(model_outputs)

    thresholds = get_thresholds(model_names=model_names)

    print("Building multimodel plots...")
    viz_outputs = make_all_multimodel_plots(
        model_outputs=model_outputs,
        thresholds=thresholds,
        outdir=outdir,
    )

    print(f"Done. Outputs saved to: {outdir}")

    return {
        "model_outputs": model_outputs,
        "thresholds": thresholds,
        "viz_outputs": viz_outputs,
        "outdir": outdir,
    }


if __name__ == "__main__":
    # In Databricks notebooks, `spark` already exists.
    # In .py jobs, this fallback creates a Spark session if needed.
    try:
        spark  # noqa: F821
    except NameError:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

    outputs = main(
        spark=spark,
        model_names=MODEL_NAMES,
        include_results=False,
        outdir="/Workspace/Users/myschne@umich.edu/CandidateSuccessModels/multimodel_viz_outputs",
    )