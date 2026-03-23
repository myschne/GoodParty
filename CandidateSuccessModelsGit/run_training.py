"""
Main training entry point for the Candidate Success modeling pipeline.

This script orchestrates the full end-to-end training workflow: it loads
training data, validates the selected model configuration, runs grouped
cross-validation, summarizes model performance, generates diagnostic plots,
fits the final model on the full dataset, and logs/registers the model in
MLflow. It serves as the top-level driver that ties together data loading,
modeling, evaluation, visualization, and experiment tracking.
"""

import os
import numpy as np
import mlflow

from config import (
    THRESHOLD,
    N_FOLDS,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    PLOT_OUTPUT_DIR,
    EXPERIMENT_PATH,
)
from load_data import load_training_data
from LogisticModelViz import make_all_plots
from mlflow_utils import (
    ensure_experiment,
    validate_model_config,
    parse_args,
)
from training_pipeline import (
    run_cross_validation,
    summarize_viability,
    summarize_feature_importance,
    fit_final_model,
    log_and_register_model,
    write_feature_catalog_to_uc
)
from modeling import build_feature_catalog

from sentiment import add_message_level_text_features
from feature_engineering import aggregate_message_level_data


# =========================================================
# Configuration
# =========================================================

mlflow.set_registry_uri("databricks-uc")

# =========================================================
# Helpers
# =========================================================

def summarize_cv_metrics(fold_aucs, cv_outputs):
    print("\n============================")
    print(f"{N_FOLDS}-Fold CV Summary")
    print("============================")
    print("Fold AUCs:", [round(a, 4) for a in fold_aucs])
    print("Mean AUC :", float(np.mean(fold_aucs)))
    print("Std AUC  :", float(np.std(fold_aucs)))
    print("\nPooled ROC-AUC:", cv_outputs["pooled_metrics"]["roc_auc"])
    print("Pooled confusion matrix:")
    print(cv_outputs["pooled_metrics"]["confusion_matrix"])
    print(cv_outputs["pooled_metrics"]["classification_report"])

# =========================================================
# Main training workflow
# =========================================================

def main(spark, model_name=None):
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    experiment_id = ensure_experiment(EXPERIMENT_PATH)
    print(f"Using MLflow experiment_id={experiment_id} at {EXPERIMENT_PATH}")

    model_spec = validate_model_config(model_name)
    model_type = model_spec["type"]
    model_params = model_spec["params"]

    print(f"Running training for model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Model params: {model_params}")

    full_df = load_training_data(spark).copy()
    print(f"Loaded raw training rows: {len(full_df)}")

    # message-level -> text features -> candidate-election level
    full_df = add_message_level_text_features(full_df)
    full_df = aggregate_message_level_data(full_df, training=True)

    print(f"Aggregated training rows: {len(full_df)}")
    print(full_df.columns.tolist())

    cv_outputs = run_cross_validation(full_df, model_type, model_params)
    fold_aucs = [m["roc_auc"] for m in cv_outputs["fold_metrics"]]

    summarize_cv_metrics(fold_aucs, cv_outputs)

    results, viability_comparison = summarize_viability(cv_outputs["oof_df"].copy())
    imp = summarize_feature_importance(cv_outputs["importance_series_list"],model_name)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    make_all_plots(
        fold_metrics_df=cv_outputs["fold_metrics_df"],
        oof_df=cv_outputs["oof_df"],
        threshold=THRESHOLD,
        results=results,
        imp=imp,
        outdir=PLOT_OUTPUT_DIR,
    )

    final_model, X_full, _ = fit_final_model(full_df, model_type, model_params)
    model_uri = log_and_register_model(
        final_model=final_model,
        X_full=X_full,
        model_name=model_name,
        model_type=model_type,
        model_params=model_params,
        fold_aucs=fold_aucs,
        pooled_metrics=cv_outputs["pooled_metrics"],
    )
    feature_catalog_df = build_feature_catalog(
        clf=final_model,
        X=X_full,
        model_name=model_name,
        model_type=model_type,
        importance_df=imp,
    )

    feature_catalog_table = write_feature_catalog_to_uc(
        spark=spark,
        feature_catalog_df=feature_catalog_df,
        model_name=model_name,
    )

    print("\nFinal model saved at:", model_uri)

    return {
        "run_info": {
            "model_name": model_name,
            "model_type": model_type,
            "model_params": model_params,
            "threshold": THRESHOLD,
            "n_folds": N_FOLDS,
        },
        "cv": {
            "folds": cv_outputs["folds"],
            "fold_metrics": cv_outputs["fold_metrics"],
            "fold_metrics_df": cv_outputs["fold_metrics_df"],
            "pooled_metrics": cv_outputs["pooled_metrics"],
        },        "predictions": {
            "oof": cv_outputs["oof_df"],
            "available_outputs": [
                col for col in ["pred_label", "pred_score", "pred_proba"]
                if col in cv_outputs["oof_df"].columns
            ],
        },
        "interpretation": {
            "feature_importance": imp,
            "importance_method": "aggregate_feature_importance" if not imp.empty else None,
            "feature_catalog": feature_catalog_df,
        },
        "artifacts": {
            "final_model": final_model,
            "model_uri": model_uri,
            "plot_output_dir": PLOT_OUTPUT_DIR,
            "feature_catalog_table": feature_catalog_table,
        },
        "diagnostics": {
            "viability_comparison": viability_comparison,
        },
    }

# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    args = parse_args()
    outputs = main(spark, model_name=args.model_name)

    print("\nTraining complete.")
    print("Saved model URI:", outputs["artifacts"]["model_uri"])
    print("\nFold metrics:")
    print(outputs["cv"]["fold_metrics_df"])