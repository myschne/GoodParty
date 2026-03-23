import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import f1_score

model_outputs = {
    "logistic_regression": {
        "fold_metrics_df": fold_metrics_lr,
        "oof_df": oof_lr,
        "results": results_lr,
        "imp": imp_lr,
    },
    "elastic_net_logistic": {
        "fold_metrics_df": fold_metrics_en,
        "oof_df": oof_en,
        "results": results_en,
        "imp": imp_en,
    },
    "random_forest": {
        "fold_metrics_df": fold_metrics_rf,
        "oof_df": oof_rf,
        "results": results_rf,
        "imp": imp_rf,
    },
    "xgboost": {
        "fold_metrics_df": fold_metrics_xgb,
        "oof_df": oof_xgb,
        "results": results_xgb,
        "imp": imp_xgb,
    },
}


def build_predictive_metrics_table(model_outputs, thresholds):
    rows = []

    for model_name, payload in model_outputs.items():
        oof_df = payload["oof_df"].copy()
        y_true = np.asarray(oof_df["y_true"]).astype(int)
        proba = np.asarray(oof_df["pred_proba"]).astype(float)
        threshold = thresholds[model_name]

        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
        })

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def build_threshold_sweep_table(model_outputs, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []

    for model_name, payload in model_outputs.items():
        oof_df = payload["oof_df"].copy()
        y_true = np.asarray(oof_df["y_true"]).astype(int)
        proba = np.asarray(oof_df["pred_proba"]).astype(float)

        auc = roc_auc_score(y_true, proba)

        for t in threshold_grid:
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

            rows.append({
                "model": model_name,
                "threshold": t,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": auc,
            })

    return pd.DataFrame(rows)

def build_bucket_calibration_table(model_outputs, bucket_col="viability_bucket_model"):
    rows = []

    for model_name, payload in model_outputs.items():
        results = payload["results"].copy()

        overall_actual_win = results["Win"].mean()

        summary = (
            results.groupby(bucket_col, dropna=False, observed=False)
            .agg(
                n=("Win", "count"),
                observed_win_rate=("Win", "mean"),
            )
            .reset_index()
        )

        summary["model"] = model_name
        summary["overall_actual_win_rate"] = overall_actual_win
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)

