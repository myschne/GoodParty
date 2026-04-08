"""
Training pipeline utilities for cross-validation, evaluation summaries,
final-model fitting, and MLflow registration.

This module contains the core orchestration helpers used by the training
workflow. It handles grouped cross-validation, pooled out-of-fold evaluation,
viability and feature-importance summaries, refitting the final model on the
full dataset, and logging/registering the resulting model in MLflow and
Unity Catalog.
"""

import os
import numpy as np
import pandas as pd
from decimal import Decimal
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


from config import (
    THRESHOLD,
    N_FOLDS,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    MODEL_CONFIGS,
    MODEL_ALIAS,
    PLOT_OUTPUT_DIR,
    UC_CATALOG,
    UC_SCHEMA,
    EXPERIMENT_PATH,
)

from cv import make_group_folds
from feature_engineering import split_X_y
from modeling import (
    make_model_pipeline,
    extract_model_importance,
    build_feature_catalog,
)
from evaluation import (
    compute_fold_metrics,
    compute_pooled_metrics,
    compare_model_vs_original_viability,
    aggregate_feature_importance,
    get_model_predictions,
    build_fold_prediction_frame,
    build_fold_metrics_df,
)

from mlflow_utils import (
    get_registered_model_name,
)

# =========================================================
# Cross-validation workflow
# =========================================================

def run_cross_validation(full_df, model_type, model_params):
    """
    Run grouped cross-validation for the selected model.

    This function:
    1. builds grouped folds
    2. fits a fresh model on each training fold
    3. scores the corresponding test fold
    4. computes fold-level metrics
    5. stores out-of-fold predictions for pooled evaluation
    6. collects fold-level feature importance summaries

    Parameters
    ----------
    full_df : pandas.DataFrame
        Candidate-level training dataset containing the target column `Win`
        plus all model features.
    model_type : str
        Model family to build, such as logistic regression or random forest.
    model_params : dict
        Hyperparameters passed into `make_model_pipeline()`.

    Returns
    -------
    dict
        Dictionary containing:
        - folds : fold definitions returned by `make_group_folds`
        - fold_metrics : list of metric dictionaries, one per fold
        - fold_metrics_df : tabular fold summary
        - oof_df : out-of-fold prediction DataFrame
        - pooled_metrics : metrics computed across all OOF predictions
        - importance_series_list : per-fold feature importance outputs
    """
    # Build grouped folds so related records stay together across train/test splits.
    folds = make_group_folds(full_df, n_splits=N_FOLDS)

    # Containers for outputs accumulated across folds.
    fold_metrics_list = []
    importance_series_list = []
    rows_all = []

    # Train and evaluate one model per fold.
    for fold_info in folds:
        fold_id = fold_info["fold"] + 1

        # Slice train/test rows for the current fold.
        train_df = full_df.iloc[fold_info["train_idx"]].copy()
        test_df = full_df.iloc[fold_info["test_idx"]].copy()

        # Split into X/y. `featured_test` preserves identifiers/metadata needed later.
        X_train, y_train, _ = split_X_y(train_df, target_col="Win")
        X_test, y_test, featured_test = split_X_y(test_df, target_col="Win")

        # Build and fit a fresh pipeline for this fold.
        clf, num_cols, cat_cols = make_model_pipeline(
            X_train,
            model_type=model_type,
            params=model_params,
        )
        clf.fit(X_train, y_train)

        # Generate standardized predictions regardless of model class.
        pred_outputs = get_model_predictions(clf, X_test, threshold=THRESHOLD)

        # Compute fold-level classification metrics.
        fold_metrics = compute_fold_metrics(
            y_true=y_test,
            pred_label=pred_outputs["pred_label"],
            pred_score=pred_outputs["pred_score"],
            fold_id=fold_id,
        )
        fold_metrics_list.append(fold_metrics)

        # Print a concise fold summary for monitoring.
        print(f"\nFold {fold_id}/{N_FOLDS}")
        print(f"ROC-AUC: {fold_metrics['roc_auc']:.4f}")
        print("Confusion matrix:")
        print(fold_metrics["confusion_matrix"])

        # Store row-level predictions so we can later evaluate all OOF predictions together.
        rows_all.append(
            build_fold_prediction_frame(
                featured_test=featured_test,
                y_true=y_test,
                pred_outputs=pred_outputs,
                fold_id=fold_id,
            )
        )

        # Extract model-specific importance/coefficient output when supported.
        importance_series = extract_model_importance(
            clf=clf,
            num_cols=num_cols,
            cat_cols=cat_cols,
            fold_name=f"fold_{fold_id}",
        )
        if importance_series is not None:
            importance_series_list.append(importance_series)

    # Concatenate all out-of-fold predictions into a single table.
    oof_df = pd.concat(rows_all, ignore_index=True)

    # Compute pooled metrics across the full set of OOF predictions.
    pooled_metrics = compute_pooled_metrics(
        y_true=oof_df["y_true"],
        pred_label=oof_df["pred_label"],
        pred_score=oof_df["pred_score"] if "pred_score" in oof_df.columns else None,
    )

    # Convert fold metric dictionaries into a reporting table.
    fold_metrics_df = build_fold_metrics_df(fold_metrics_list)

    return {
        "folds": folds,
        "fold_metrics": fold_metrics_list,
        "fold_metrics_df": fold_metrics_df,
        "oof_df": oof_df,
        "pooled_metrics": pooled_metrics,
        "importance_series_list": importance_series_list,
    }


# =========================================================
# Reporting helpers
# =========================================================

def summarize_viability(results):
    """
    Compare model-derived viability buckets to the original viability labels.

    This function expects a prediction DataFrame, typically the out-of-fold
    prediction table produced during cross-validation. It adds standardized
    probability and predicted-label columns, runs the viability comparison,
    prints several diagnostics, and returns the enriched results.

    Parameters
    ----------
    results : pandas.DataFrame
        Prediction-level DataFrame. Must include `pred_label`, and for full
        viability analysis it must also include `pred_proba`.

    Returns
    -------
    tuple
        (results, viability_comparison)

        results : pandas.DataFrame
            Input DataFrame enriched with model-derived viability outputs.

        viability_comparison : dict or None
            Summary produced by `compare_model_vs_original_viability`, or
            None if probabilities are unavailable.
    """
    # Viability mapping requires predicted probabilities.
    if "pred_proba" not in results.columns:
        print("\nSkipping viability comparison: model did not produce probability outputs.")
        return results, None

    results = results.copy()

    # Standardize column names expected by the viability comparison logic.
    results["proba_win"] = results["pred_proba"]
    results["pred_win"] = results["pred_label"]

    viability_comparison = compare_model_vs_original_viability(results)
    results = viability_comparison["results"]

    # Print diagnostic summaries to compare original vs model-based viability.
    print("\nViability bucket distribution (model):")
    print(results["viability_bucket_model"].value_counts(dropna=False))

    print("\nCrosstab: original viability bucket (rows) vs model bucket (cols)")
    print(viability_comparison["crosstab"])

    print("\nExact bucket agreement:", round(viability_comparison["exact_agreement"], 4))
    print("\nBucket distance distribution:")
    print(viability_comparison["bucket_distance_distribution"])
    print("\nMean bucket distance:", round(viability_comparison["mean_bucket_distance"], 4))
    print(
        "\nCorrelation between original viability_score and model-derived viability_score:",
        round(viability_comparison["score_correlation"], 4)
        if pd.notna(viability_comparison["score_correlation"])
        else np.nan,
    )

    return results, viability_comparison


def summarize_feature_importance(importance_series_list, model_name):
    """
    Aggregate and print feature importance across folds.

    Works for models where fold-level importance or coefficient values can be
    extracted. For linear models, this may include signed coefficient summaries.
    For tree-based models, the aggregation depends on how `extract_model_importance`
    is implemented.

    Parameters
    ----------
    importance_series_list : list
        List of per-fold feature importance outputs.

    Returns
    -------
    pandas.DataFrame
        Aggregated feature importance table. Returns an empty DataFrame if
        no importance information is available.
    """
    if not importance_series_list:
        print("\nNo model-specific importance values were returned; skipping importance summaries.")
        return pd.DataFrame()

    # Aggregate feature importance values across all folds.
    signed = model_name in {
    "logistic_regression",
    "elastic_net_logistic",
    "lasso_logistic",
    "ridge_logistic",
    }
 
    imp = aggregate_feature_importance(importance_series_list, signed=signed)

    print("\nTop 25 features by mean absolute importance across folds:")
    print(imp.head(25).to_string())

    # If coefficient-style summaries are available, also show directionality.
    if signed and "mean_importance" in imp.columns:
        print("\nTop 25 features pushing toward Win=1 on average:")
        print(
            imp.sort_values("mean_importance", ascending=False)
            .head(25)[["mean_importance", "std_importance", "mean_abs_importance", "sign_consistency"]]
            .to_string()
        )

        print("\nTop 25 features pushing toward Win=0 on average:")
        print(
            imp.sort_values("mean_importance", ascending=True)
            .head(25)[["mean_importance", "std_importance", "mean_abs_importance", "sign_consistency"]]
            .to_string()
        )

    return imp


# =========================================================
# Final model training
# =========================================================

def fit_final_model(full_df, model_type, model_params):
    """
    Fit the final model on the full training dataset.

    After cross-validation is complete, this function retrains the model on
    all available labeled data so that the final registered model uses the
    maximum amount of information.

    Parameters
    ----------
    full_df : pandas.DataFrame
        Full labeled training dataset.
    model_type : str
        Model family to build.
    model_params : dict
        Hyperparameters passed into `make_model_pipeline()`.

    Returns
    -------
    tuple
        (final_model, X_full, y_full)

        final_model : fitted estimator/pipeline
            Model trained on all available rows.
        X_full : pandas.DataFrame
            Final training features.
        y_full : pandas.Series or numpy.ndarray
            Final training target.
    """
    X_full, y_full, _ = split_X_y(full_df, target_col="Win")
    final_model, _, _ = make_model_pipeline(
        X_full,
        model_type=model_type,
        params=model_params,
    )
    final_model.fit(X_full, y_full)
    return final_model, X_full, y_full


# =========================================================
# MLflow logging and registration
# =========================================================

def log_and_register_model(
    final_model,
    X_full,
    model_name,
    model_type,
    model_params,
    fold_aucs,
    pooled_metrics,
):
    """
    Log the final model to MLflow and register it in Unity Catalog.

    This function:
    1. builds a model signature from sample inputs/outputs
    2. logs run parameters and evaluation metrics
    3. logs the sklearn model artifact
    4. registers the model in UC-backed MLflow registry
    5. assigns the configured model alias

    Parameters
    ----------
    final_model : fitted estimator/pipeline
        Final model trained on all labeled data.
    X_full : pandas.DataFrame
        Full training feature matrix used for generating the input example
        and model signature.
    model_name : str
        Logical model name from config.
    model_type : str
        Model family name, such as logistic regression or random forest.
    model_params : dict
        Hyperparameters used to fit the final model.
    fold_aucs : list[float]
        Fold-level ROC-AUC values from cross-validation.
    pooled_metrics : dict
        Pooled evaluation metrics from out-of-fold predictions.

    Returns
    -------
    str
        MLflow model URI for the logged model artifact.
    """
    # Build the fully qualified UC model name.
    registered_model_name = get_registered_model_name(model_name)

    # Use a small feature sample to infer the model signature.
    input_example = X_full.head(5)
    final_outputs = get_model_predictions(final_model, input_example, threshold=THRESHOLD)

    # Choose the most informative output available for signature inference.
    if final_outputs["pred_proba"] is not None:
        output_example = final_outputs["pred_proba"]
    elif final_outputs["pred_score"] is not None:
        output_example = final_outputs["pred_score"]
    else:
        output_example = final_outputs["pred_label"]

    signature = infer_signature(model_input=input_example, model_output=output_example)

    # Summarize CV performance for logging.
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    # Start an MLflow run and log model metadata.
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("threshold", THRESHOLD)

        # Log all model hyperparameters under a consistent prefix.
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_param__{param_name}", param_value)

        # Log cross-validation and pooled OOF performance metrics.
        mlflow.log_metric("mean_auc", mean_auc)
        mlflow.log_metric("std_auc", std_auc)
        mlflow.log_metric("pooled_auc", float(pooled_metrics["roc_auc"]))
        mlflow.log_metric("accuracy", float(pooled_metrics["accuracy"]))
        mlflow.log_metric("precision", float(pooled_metrics["precision"]))
        mlflow.log_metric("recall", float(pooled_metrics["recall"]))
        mlflow.log_metric("f1_score", float(pooled_metrics["f1_score"]))

        # Log and register the final sklearn model.

        input_example = X_full.head(5).copy()

        for col in input_example.columns:
            input_example[col] = input_example[col].apply(
                lambda x: float(x) if isinstance(x, Decimal) else x
            )

        input_example = input_example.infer_objects(copy=False)

        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature,
        )

    # Assign the configured alias (for example, champion/candidate/current).
    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=MODEL_ALIAS,
        version=model_info.registered_model_version,
    )

    return model_info.model_uri

# =========================================================
# Unity Catalog table helpers
# =========================================================

def get_feature_catalog_table_name(model_name: str) -> str:
    """
    Build the fully qualified Unity Catalog table name for the feature catalog.

    Parameters
    ----------
    model_name : str
        Logical model name used throughout the project.

    Returns
    -------
    str
        Fully qualified table name for storing the model's feature catalog.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_feature_catalog"


def write_feature_catalog_to_uc(spark, feature_catalog_df, model_name):
    """
    Write the model feature catalog to a Unity Catalog table.

    This helper converts the pandas feature catalog into a Spark DataFrame,
    enforces stable numeric types for importance columns, overwrites the
    target table, and returns the written table name.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session used to write the table.
    feature_catalog_df : pandas.DataFrame
        Feature catalog dataframe describing model inputs and, when available,
        feature importance values.
    model_name : str
        Logical model name used to build the target table name.

    Returns
    -------
    str
        Fully qualified Unity Catalog table name written by this function.
    """
    output_table = get_feature_catalog_table_name(model_name)

    df = feature_catalog_df.copy()

    # Force stable numeric schema for Delta / Spark
    for col in ["importance", "importance_abs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    spark_df = spark.createDataFrame(df)
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)

    print(f"Wrote feature catalog table to: {output_table}")
    return output_table


def write_df_to_uc(spark, df, table_name):
    """
    Write a pandas DataFrame to a Unity Catalog table.

    This helper provides a generic path for persisting pandas outputs from
    the training workflow, such as out-of-fold predictions or fold metrics,
    into overwrite-mode Spark tables.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session used to write the table.
    df : pandas.DataFrame
        Dataframe to convert to Spark and persist.
    table_name : str
        Fully qualified Unity Catalog table name.

    Returns
    -------
    str
        The table name that was written.
    """
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    return table_name


def get_oof_table(model_name):
    """
    Build the fully qualified Unity Catalog table name for out-of-fold predictions.

    Parameters
    ----------
    model_name : str
        Logical model name used throughout the project.

    Returns
    -------
    str
        Fully qualified table name for storing out-of-fold prediction outputs.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_oof_predictions"


def get_fold_metrics_table(model_name):
    """
    Build the fully qualified Unity Catalog table name for fold-level metrics.

    Parameters
    ----------
    model_name : str
        Logical model name used throughout the project.

    Returns
    -------
    str
        Fully qualified table name for storing cross-validation fold metrics.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_fold_metrics"