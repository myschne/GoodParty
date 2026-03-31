"""
Preprocessing utilities for model training.

This module builds a sklearn ColumnTransformer that applies separate
transformations to numeric and categorical features.

Current preprocessing behavior:
- Numeric columns:
    - median imputation
    - optional standardization for models that benefit from scaling
- Categorical columns:
    - missing-value fill with "Unknown"
    - one-hot encoding with unseen-category protection

The function returns both the fitted preprocessing object definition and
the column lists used to build it. Returning the column lists is useful
for debugging, feature audits, and downstream interpretation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X, model_type=None) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a preprocessing pipeline for numeric and categorical features.

    The function automatically identifies numeric and categorical columns
    from the input feature matrix `X` and creates a ColumnTransformer with
    separate pipelines for each type.

    Preprocessing logic
    -------------------
    Numeric columns:
    - Impute missing values using the median
    - Apply StandardScaler unless the model type is "random_forest"

    Categorical columns:
    - Impute missing values with the constant value "Unknown"
    - One-hot encode categories
    - Ignore unseen categories at prediction time

    Why scaling is skipped for random forests
    -----------------------------------------
    Tree-based models such as random forests do not rely on feature scale,
    so standardization is unnecessary. Linear models and distance-based
    models usually benefit from scaling, so it is included by default.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix containing both numeric and categorical columns.
    model_type : str, optional
        Model family used to determine whether numeric scaling should be
        applied. If set to "random_forest", scaling is skipped.

    Returns
    -------
    tuple
        A tuple containing:
        - preprocessor : sklearn.compose.ColumnTransformer
            Combined preprocessing transformer
        - num_cols : list[str]
            Names of numeric columns
        - cat_cols : list[str]
            Names of categorical columns
    """
    # Identify numeric and categorical columns automatically from dtypes.
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Tree-based models do not require scaling, so use only imputation.
    if model_type == "random_forest":
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
    else:
        # For models like logistic regression, scaling helps place numeric
        # variables on a comparable scale after imputation.
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    # Fill missing categorical values with a placeholder, then one-hot encode.
    # handle_unknown="ignore" prevents errors if new categories appear at
    # scoring time that were not seen during training.
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Apply the numeric and categorical pipelines to their respective columns.
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    return preprocessor, num_cols, cat_cols