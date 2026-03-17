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