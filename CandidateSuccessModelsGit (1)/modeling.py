"""
Model-building utilities for estimator construction, pipeline assembly,
and feature-importance extraction.

This module centralizes the logic for creating supported scikit-learn
estimators, combining them with preprocessing into a full training pipeline,
and extracting model coefficients or feature importances after fitting.
It keeps model-specific setup separate from the main training workflow so
the pipeline remains modular, reusable, and easy to extend.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from config import MAX_ITER
from preprocessing import make_preprocessor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def build_estimator(model_type, params):
    """
    Build and return the estimator associated with the requested model type.

    Parameters
    ----------
    model_type : str
        Name of the model family to instantiate.
        Supported values currently include:
        - "logistic_regression"
        - "elastic_net_logistic"
        - "random_forest"
    params : dict
        Hyperparameters to pass into the estimator constructor.

    Returns
    -------
    estimator
        An unfit scikit-learn estimator.

    Raises
    ------
    ValueError
        If model_type is not supported.

    Notes
    -----
    - Standard logistic regression and elastic net logistic regression
      both use sklearn.linear_model.LogisticRegression, but with
      different parameter settings.
    - Additional model types can be added here as the project expands.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(**params)

    if model_type == "elastic_net_logistic":
        return LogisticRegression(**params)

    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    
    if model_type == "xgboost":
        return XGBClassifier(**params)
    
    # Add more models here

    raise ValueError(f"Unsupported model_type: {model_type}")


def make_model_pipeline(X, model_type, params):
    """
    Create a full modeling pipeline consisting of preprocessing plus estimator.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix used to infer numeric and categorical columns
        for preprocessing.
    model_type : str
        Name of the model family to build.
    params : dict
        Hyperparameters passed to the selected estimator.

    Returns
    -------
    tuple
        (clf, num_cols, cat_cols)

        clf : sklearn.pipeline.Pipeline
            Full modeling pipeline with:
            - "preprocess": feature preprocessing transformer
            - "model": estimator
        num_cols : list
            Numeric columns identified by the preprocessor builder.
        cat_cols : list
            Categorical columns identified by the preprocessor builder.

    Notes
    -----
    This function keeps model training generic by separating:
    - preprocessing logic
    - estimator construction
    - pipeline assembly
    """
    # Build preprocessing pipeline and record feature groups.
    preprocessor, num_cols, cat_cols = make_preprocessor(X, model_type=model_type)

    # Build the requested estimator.
    estimator = build_estimator(model_type, params)

    # Combine preprocessing and model into one sklearn Pipeline.
    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    return clf, num_cols, cat_cols


def extract_model_importance(clf, num_cols, cat_cols, fold_name):
    """
    Extract feature importance or coefficients from a fitted pipeline.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted modeling pipeline containing:
        - a "preprocess" step
        - a "model" step
    num_cols : list
        Original numeric feature names passed into preprocessing.
    cat_cols : list
        Original categorical feature names passed into preprocessing.
    fold_name : str
        Name to assign to the returned Series, typically identifying the fold
        from which the model was fit.

    Returns
    -------
    pd.Series or None
        Feature importance series indexed by transformed feature names,
        or None if the estimator does not expose coefficients or feature
        importances.

    Notes
    -----
    Feature names are reconstructed by combining:
    - numeric column names
    - one-hot encoded categorical feature names from the fitted encoder

    Supported model outputs:
    - coef_ for linear models such as logistic regression
    - feature_importances_ for tree-based models such as random forest
    """
    # Pull fitted preprocessing and model steps from the pipeline.
    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # Recover expanded one-hot encoded categorical feature names.
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols)

    # Combine numeric and encoded categorical names in transformed order.
    feature_names = np.r_[num_cols, cat_names]

    # Linear-model coefficients
    if hasattr(model, "coef_"):
        return pd.Series(model.coef_.ravel(), index=feature_names, name=fold_name)

    # Tree-based feature importances
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names, name=fold_name)

    return None

def build_feature_catalog(clf, X, model_name, model_type, importance_df=None):
    """
    Build a feature catalog for the fitted model, including:
    - raw input features before preprocessing
    - transformed features actually seen by the estimator
    - feature types
    - importance / coefficient values when available
    """
    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # Raw input feature types from X
    raw_num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    raw_cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    raw_rows = []
    for col in raw_num_cols:
        raw_rows.append({
            "model_name": model_name,
            "model_type": model_type,
            "feature_stage": "raw_model_input",
            "feature_name": col,
            "source_feature": col,
            "value_type": "numeric",
            "importance": None,
            "importance_abs": None,
            "importance_kind": "not_available",
        })

    for col in raw_cat_cols:
        raw_rows.append({
            "model_name": model_name,
            "model_type": model_type,
            "feature_stage": "raw_model_input",
            "feature_name": col,
            "source_feature": col,
            "value_type": "categorical",
            "importance": None,
            "importance_abs": None,
            "importance_kind": "not_available",
        })

    # Transformed feature names
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(raw_cat_cols).tolist()
    transformed_feature_names = raw_num_cols + cat_feature_names

    transformed_rows = []

    # Numeric transformed features
    for col in raw_num_cols:
        transformed_rows.append({
            "model_name": model_name,
            "model_type": model_type,
            "feature_stage": "transformed_model_input",
            "feature_name": col,
            "source_feature": col,
            "value_type": "numeric",
            "importance": None,
            "importance_abs": None,
            "importance_kind": "not_available",
        })

    # One-hot encoded categorical transformed features
    for feat in cat_feature_names:
        source_col = None
        for c in raw_cat_cols:
            prefix = f"{c}_"
            if feat.startswith(prefix):
                source_col = c
                break

        transformed_rows.append({
            "model_name": model_name,
            "model_type": model_type,
            "feature_stage": "transformed_model_input",
            "feature_name": feat,
            "source_feature": source_col,
            "value_type": "onehot_encoded_categorical",
            "importance": None,
            "importance_abs": None,
            "importance_kind": "not_available",
        })

    out = pd.DataFrame(transformed_rows)

    # Attach importance if available
    if importance_df is not None and not importance_df.empty:
        imp = importance_df.copy().reset_index().rename(columns={"index": "feature_name"})

        if "mean_importance" in imp.columns:
            imp["importance"] = imp["mean_importance"]
            imp["importance_abs"] = imp.get("mean_abs_importance", imp["importance"].abs())
            imp["importance_kind"] = "coefficient"
        elif "mean_abs_importance" in imp.columns:
            # defensive fallback
            imp["importance"] = imp["mean_abs_importance"]
            imp["importance_abs"] = imp["mean_abs_importance"]
            imp["importance_kind"] = "feature_importance"
        else:
            imp["importance"] = None
            imp["importance_abs"] = None
            imp["importance_kind"] = "not_available"

        # For tree models, aggregated importance may already be in mean_importance too,
        # so use model family to label it more clearly.
        if model_type in {"random_forest", "xgboost"}:
            imp["importance_kind"] = "feature_importance"

        out = out.merge(
            imp[["feature_name", "importance", "importance_abs", "importance_kind"]],
            on="feature_name",
            how="left",
            suffixes=("", "_imp"),
        )

        # fill from merged columns where applicable
        for col in ["importance", "importance_abs", "importance_kind"]:
            imp_col = f"{col}_imp"
            if imp_col in out.columns:
                out[col] = out[imp_col].combine_first(out[col])
                out = out.drop(columns=[imp_col])

    return out