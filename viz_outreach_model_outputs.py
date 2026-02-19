# viz_outreach_model_outputs.py
# Visualizations for the outputs produced by your training/eval script.
#
# Assumptions:
# - Your existing script produces these in-memory objects:
#     fold_aucs (list[float])
#     y_all (np.ndarray)
#     proba_all (np.ndarray)
#     pred_all (np.ndarray)  # optional; recomputed if missing
#     THRESHOLD (float)
#     results (pd.DataFrame) with columns:
#         - proba_win, pred_win
#         - viability_score_model, viability_bucket_model, viability_bucket_num
#         - viability_score_mean, viability_bucket_orig, viability_bucket_orig_num
#         - bucket_distance
#     imp (pd.DataFrame) index=feature, columns:
#         mean_coef, std_coef, mean_abs_coef, sign_consistency
#
# If you prefer file-based I/O instead, see the "EXPORT/IMPORT" notes at bottom.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

# -----------------------
# Small utilities
# -----------------------

def _safe_pred_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)

def _confusion_rates(cm: np.ndarray):
    # cm = [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan  # recall
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan  # specificity
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    return {"TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr}

def _maybe_int_series(s):
    # Helpful if bucket labels are categorical and you want stable ordering
    return s.astype("Int64") if pd.api.types.is_integer_dtype(s) or str(s.dtype) == "Int64" else s


# -----------------------
# Plotting functions
# -----------------------

def plot_fold_aucs(fold_aucs, savepath=None):
    fig = plt.figure()
    x = np.arange(1, len(fold_aucs) + 1)
    plt.plot(x, fold_aucs, marker="o")
    plt.axhline(np.mean(fold_aucs), linestyle="--")
    plt.xticks(x)
    plt.xlabel("Fold")
    plt.ylabel("ROC-AUC")
    plt.title("Per-fold ROC-AUC")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_roc_curve(y_true, proba, savepath=None):
    fig = plt.figure()
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pooled)")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_pr_curve(y_true, proba, savepath=None):
    fig = plt.figure()
    p, r, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.plot(r, p, label=f"Avg Precision={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Pooled)")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_score_hist_by_class(y_true, proba, threshold, savepath=None):
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
    fig = plt.figure()
    pred = _safe_pred_from_proba(proba, threshold)
    cm = confusion_matrix(y_true, pred)
    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    plt.imshow(cm_display, aspect="auto")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))

    # annotate
    for (i, j), v in np.ndenumerate(cm_display):
        txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig, cm, _confusion_rates(cm)

def plot_threshold_sweep(y_true, proba, savepath=None):
    # plots precision/recall/f1 and TPR/FPR across thresholds
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s, tprs, fprs = [], [], [], [], []

    from sklearn.metrics import precision_score, recall_score, f1_score

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
    # Requires results["viability_bucket_orig"] and results["viability_bucket_model"]
    fig = plt.figure()

    orig = results["viability_bucket_orig"].astype(str).value_counts().sort_index()
    model = results["viability_bucket_model"].astype(str).value_counts().sort_index()

    # align indices
    idx = sorted(set(orig.index).union(set(model.index)))
    orig = orig.reindex(idx, fill_value=0)
    model = model.reindex(idx, fill_value=0)

    x = np.arange(len(idx))
    width = 0.4
    plt.bar(x - width/2, orig.values, width=width, label="Original")
    plt.bar(x + width/2, model.values, width=width, label="Model")
    plt.xticks(x, idx, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Viability bucket distribution: Original vs Model")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_bucket_distance(results: pd.DataFrame, savepath=None):
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
    fig = plt.figure()
    ct = pd.crosstab(results["viability_bucket_orig"], results["viability_bucket_model"], dropna=False)
    plt.imshow(ct.values, aspect="auto")
    plt.xticks(np.arange(ct.shape[1]), ct.columns.astype(str), rotation=30, ha="right")
    plt.yticks(np.arange(ct.shape[0]), ct.index.astype(str))
    plt.xlabel("Model bucket")
    plt.ylabel("Original bucket")
    plt.title("Crosstab heatmap: Original vs Model viability buckets")

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            plt.text(j, i, str(ct.iloc[i, j]), ha="center", va="center")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_top_features(imp: pd.DataFrame, top_n=25, which="mean_abs_coef", savepath=None):
    # which in {"mean_abs_coef","mean_coef","std_coef","sign_consistency"}
    fig = plt.figure()

    s = imp[which].sort_values(ascending=False).head(top_n)
    # horizontal bar is easier to read
    plt.barh(np.arange(len(s)), s.values)
    plt.yticks(np.arange(len(s)), s.index)
    plt.gca().invert_yaxis()
    plt.xlabel(which)
    plt.title(f"Top {top_n} features by {which}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_top_pos_neg_features(imp: pd.DataFrame, top_n=20, savepath_pos=None, savepath_neg=None):
    # two plots: most positive / most negative mean_coef
    fig1 = plt.figure()
    pos = imp["mean_coef"].sort_values(ascending=False).head(top_n)
    plt.barh(np.arange(len(pos)), pos.values)
    plt.yticks(np.arange(len(pos)), pos.index)
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} positive mean coefficients (toward Win=1)")
    plt.tight_layout()
    if savepath_pos:
        plt.savefig(savepath_pos, dpi=200)

    fig2 = plt.figure()
    neg = imp["mean_coef"].sort_values(ascending=True).head(top_n)
    plt.barh(np.arange(len(neg)), neg.values)
    plt.yticks(np.arange(len(neg)), neg.index)
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} negative mean coefficients (toward Win=0)")
    plt.tight_layout()
    if savepath_neg:
        plt.savefig(savepath_neg, dpi=200)

    return fig1, fig2
def plot_metrics_table(y_true: np.ndarray, proba: np.ndarray, threshold: float, outdir: Path) -> None:
    outdir = Path(outdir)            
    outdir.mkdir(parents=True, exist_ok=True)

    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else np.nan  # sensitivity
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan  # specificity
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    # Build a compact display table
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

    # “Pretty” formatting
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.35)

    # header style
    for j in range(2):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.15)

    # subtle row striping (keeps default matplotlib colors)
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(2):
                tbl[i, j].set_alpha(0.85)

    fig.tight_layout()
    fig.savefig(outdir / "metrics_table.png", dpi=200)
    plt.close(fig)



# -----------------------
# Main entry point
# -----------------------

def make_all_plots(
    fold_aucs,
    y_all,
    proba_all,
    threshold,
    results: pd.DataFrame,
    imp: pd.DataFrame,
    outdir="viz_outputs",
):
    import os
    os.makedirs(outdir, exist_ok=True)

    # Ensure arrays
    y_all = np.asarray(y_all)
    proba_all = np.asarray(proba_all)


    # 1) Fold AUCs
    plot_fold_aucs(fold_aucs, savepath=f"{outdir}/fold_aucs.png")

    # 2) ROC + PR
    plot_roc_curve(y_all, proba_all, savepath=f"{outdir}/roc_curve.png")
    plot_pr_curve(y_all, proba_all, savepath=f"{outdir}/pr_curve.png")

    # 3) Score distributions
    plot_score_hist_by_class(y_all, proba_all, threshold, savepath=f"{outdir}/score_hist_by_class.png")

    # 4) Confusion matrices
    _, cm, rates = plot_confusion_matrix(y_all, proba_all, threshold, normalize=False, savepath=f"{outdir}/confusion_matrix.png")
    plot_confusion_matrix(y_all, proba_all, threshold, normalize=True, savepath=f"{outdir}/confusion_matrix_normalized.png")

    # Save a small text summary for CM rates
    with open(f"{outdir}/confusion_rates.txt", "w") as f:
        f.write("Confusion matrix (tn fp / fn tp):\n")
        f.write(str(cm) + "\n\n")
        f.write("Rates:\n")
        for k, v in rates.items():
            f.write(f"{k}: {v:.4f}\n")

    # 5) Threshold sweep
    plot_threshold_sweep(y_all, proba_all, savepath=f"{outdir}/threshold_sweep.png")

    # 6) Viability bucket comparisons
    required_cols = {"viability_bucket_orig", "viability_bucket_model", "bucket_distance"}
    if required_cols.issubset(set(results.columns)):
        plot_viability_bucket_distributions(results, savepath=f"{outdir}/bucket_distribution.png")
        plot_bucket_crosstab_heatmap(results, savepath=f"{outdir}/bucket_crosstab_heatmap.png")
        plot_bucket_distance(results, savepath=f"{outdir}/bucket_distance.png")

    # 7) Feature importance plots
    if imp is not None and len(imp) > 0:
        plot_top_features(imp, top_n=25, which="mean_abs_coef", savepath=f"{outdir}/top_features_mean_abs_coef.png")
        plot_top_features(imp, top_n=25, which="sign_consistency", savepath=f"{outdir}/top_features_sign_consistency.png")
        plot_top_pos_neg_features(
            imp,
            top_n=20,
            savepath_pos=f"{outdir}/top_positive_mean_coef.png",
            savepath_neg=f"{outdir}/top_negative_mean_coef.png"
        )
    
    # 8) Metrics table
    plot_metrics_table(y_all, proba_all, threshold, outdir)
    
    

    # Close figures to avoid memory accumulation in notebooks
    plt.close("all")


if __name__ == "__main__":
    # -----------------------
    # HOW TO USE
    # -----------------------
    # Option 1 (recommended): import this file from your training script AFTER it finishes,
    # then call make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp).
    #
    # Example at the very bottom of your training file:
    #   from viz_outreach_model_outputs import make_all_plots
    #   make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp, outdir="viz_outputs")
    #
    # Option 2: If you export artifacts to disk (results.csv, imp.csv, arrays),
    # you can load them here and run. Uncomment below and point to your files.

    # ---- EXPORT/IMPORT option (uncomment if you want standalone) ----
    # import os
    # ARTIFACT_DIR = "artifacts"
    # fold_aucs = np.load(f"{ARTIFACT_DIR}/fold_aucs.npy").tolist()
    # y_all = np.load(f"{ARTIFACT_DIR}/y_all.npy")
    # proba_all = np.load(f"{ARTIFACT_DIR}/proba_all.npy")
    # THRESHOLD = float(open(f"{ARTIFACT_DIR}/threshold.txt").read().strip())
    # results = pd.read_csv(f"{ARTIFACT_DIR}/results.csv")
    # imp = pd.read_csv(f"{ARTIFACT_DIR}/imp.csv", index_col=0)
    #
    # make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp, outdir="viz_outputs")

    pass
