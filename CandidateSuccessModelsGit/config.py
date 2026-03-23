"""
Central configuration file for the Candidate Success modeling pipeline.

This file stores the shared settings used across training, evaluation,
visualization, model registration, and scoring. Keeping these values in one
place makes the pipeline easier to maintain, keeps behavior consistent across
scripts, and allows model runs to be reproduced more reliably.

This configuration covers:
- MLflow experiment tracking
- modeling and evaluation settings
- active model selection
- output paths
- Unity Catalog registration settings
- supported model configurations
- viability label definitions
- columns excluded from modeling features

In practice, this file acts as the single source of truth for project-level
constants. Other modules import from this file rather than hard-coding values
such as thresholds, model names, registry paths, or dropped columns.
"""

# =========================================================
# MLflow / experiment tracking configuration
# =========================================================

# MLflow experiment path used to store runs, metrics, parameters,
# artifacts, and model registration history for this project.
EXPERIMENT_PATH = "/Shared/CandidateSuccessModels"

# =========================================================
# Modeling and evaluation settings
# =========================================================

# Probability threshold used to convert predicted probabilities
# into binary class predictions.
#
# Example:
# - predicted probability >= 0.55 -> predict Win = 1
# - predicted probability <  0.55 -> predict Win = 0
THRESHOLD = 0.8

# Exponential decay rate for outreach recency weighting.
# Higher values would place more weight on recent outreach
# and less weight on older outreach.
#
# This is only relevant if recency-weighted features are used
# in feature engineering.
ALPHA = 0.1

# Number of grouped cross-validation folds used during training.
# Candidate groups are split across these folds so that grouped
# observations stay together during validation.
N_FOLDS = 5

# Random seed for reproducibility across model training steps
# that involve randomness, such as train/test splitting or
# random forest estimation.
RANDOM_STATE = 42

# Maximum number of optimization iterations for models that use
# iterative solvers, such as logistic regression or elastic net.
MAX_ITER = 2000

# =========================================================
# Active model selection
# =========================================================

# Name of the default model configuration to use for the current run.
# This key must match one of the entries in MODEL_CONFIGS below.
# Valid options:
# - "logistic_regression"
# - "elastic_net_logistic"
# - "random_forest"
MODEL_NAME = "xgboost"

# Alias assigned to the currently promoted production model
# in MLflow Model Registry / Unity Catalog.
#
# Example usage:
# scoring code can load models:/<registered_model_name>@champion
# so that it always uses the currently promoted model version.
MODEL_ALIAS = "champion"

# =========================================================
# Output paths
# =========================================================

# Directory where diagnostic plots and evaluation visualizations
# are saved after training.
PLOT_OUTPUT_DIR = "/Workspace/Users/myschne@umich.edu/CandidateSuccessModels/viz_outputs"

# =========================================================
# Unity Catalog registration settings
# =========================================================

# Unity Catalog catalog where registered models and scored outputs
# are stored.
UC_CATALOG = "goodparty_data_catalog"

# Unity Catalog schema used for model registration and prediction
# output tables.
UC_SCHEMA = "models_mban"

# =========================================================
# Model configuration registry
# =========================================================
# This dictionary defines all supported model options for the project.
#
# Structure:
# MODEL_CONFIGS = {
#     "model_key": {
#         "type": "<internal model type name>",
#         "params": {<model hyperparameters>}
#     }
# }
#
# The selected MODEL_NAME must match one of these top-level keys.
# Each entry is used by the training pipeline to:
# - choose which estimator to build
# - pass model-specific hyperparameters into the pipeline
# - keep the training workflow generic across model types
# =========================================================


MODEL_CONFIGS = {
    "logistic_regression": {
        # Standard logistic regression classifier
        "type": "logistic_regression",
        "params": {
            # Maximum number of solver iterations
            "max_iter": MAX_ITER,

            # Random seed for reproducibility
            "random_state": RANDOM_STATE,

            # Inverse regularization strength
            # Smaller C = stronger regularization
            "C": 0.1,

            #More weight to the minority class and less weight to the majority class.
            "class_weight": "balanced",
        },
    },

    "elastic_net_logistic": {
        # Logistic regression with elastic net regularization
        # Combines L1 and L2 penalties to support both shrinkage
        # and feature selection.
        "type": "elastic_net_logistic",
        "params": {
            # Solver required for elastic net in scikit-learn
            "solver": "saga",

            # Penalty type
            "penalty": "elasticnet",

            # Mix between L1 and L2 regularization
            # 0.0 = pure L2, 1.0 = pure L1
            "l1_ratio": 0.2,

            # Inverse regularization strength
            # Smaller C = stronger regularization
            "C": 0.1,

            # Maximum number of solver iterations
            "max_iter": MAX_ITER,

            # Random seed for reproducibility
            "random_state": RANDOM_STATE,

        },
    },
    "random_forest": {
        # Random forest classifier
        # Useful for nonlinear patterns and interaction effects
        # without manually specifying interactions.
        "type": "random_forest",
        "params": {
            # Number of trees in the forest
            "n_estimators": 300,

            # Maximum depth of each tree
            # None means nodes expand until stopping rules are met
            "max_depth": None,

            # Minimum samples required to split an internal node
            "min_samples_split": 2,

            # Minimum samples required at a leaf node
            "min_samples_leaf": 1,

            # Whether each tree is fit on a bootstrap sample
            "bootstrap": True,

            # Use all available CPU cores
            "n_jobs": -1,

            # Random seed for reproducibility
            "random_state": RANDOM_STATE,
        },
    },

    "xgboost": {
        # XGBoost classifier
        # Gradient-boosted trees are useful for capturing nonlinear
        # relationships and feature interactions with strong predictive performance.
        "type": "xgboost",
        "params": {
            # Number of boosting rounds / trees
            "n_estimators": 200,
            # Maximum depth of each tree
            # Higher values increase model complexity
            "max_depth": 6,
            # Step size shrinkage applied at each boosting round
            # Smaller values are more conservative and often require more trees
            "learning_rate": 0.03,
            # Fraction of training rows sampled for each tree
            # Helps reduce overfitting
            "subsample": 0.8,
            # Fraction of features sampled for each tree
            # Helps reduce overfitting and improve generalization
            "colsample_bytree": 0.8,
            # L1 regularization strength on leaf weights
            "reg_alpha": 0.1,
            # L2 regularization strength on leaf weights
            "reg_lambda": 1.0,
             # Random seed for reproducibility
            "random_state": RANDOM_STATE,
            # Use all available CPU cores
            "n_jobs": -1,
            # Evaluation metric for binary classification training
            "eval_metric": "logloss",
        },
    },
}

# =========================================================
# Viability label mapping
# =========================================================

VIAB_LABELS = [
    # Ordered labels used to translate model-derived or original
    # viability scores into human-readable categories.
    #
    # Intended progression:
    # 0 -> No Chance
    # 1 -> Unlikely to Win
    # 2 -> Has a Chance
    # 3 -> Likely to Win
    # 4 -> Frontrunner
    "No Chance",
    "Unlikely to Win",
    "Has a Chance",
    "Likely to Win",
    "Frontrunner"
]

# =========================================================
# Columns excluded from model training features
# =========================================================
# These columns are dropped before modeling because they fall into
# one of these buckets:
# - target / outcome leakage
# - identifiers
# - prior scoring variables
# - raw fields replaced by engineered features
# - fields not currently modeled
#
# Important:
# Many of these columns are intentionally replaced by transformed
# versions created later in feature engineering.
# =========================================================

DROP_COLS = [
    # Outcome Columns
    # Removed to avoid target leakage, since these directly contain
    # election outcome information or post-election results.
    "Win",
    "general_election_result",
    "general_votes_received",
    "total_general_votes_cast",

    # Identifier
    # Unique candidate identifier; useful for joins and grouping,
    # but should not be used as a predictive feature.
    "hubspot_id",

    # Previous Model Scores
    # Dropped to prevent the new model from learning directly from
    # prior viability score outputs rather than underlying features.
    "viability_score_mean",
    "viability_score_max",
    
    # Re-Engineered Features
    # Raw source columns removed because they are transformed into
    # more useful modeling features downstream.
    "election_date", 
    "latest_outreach", # -> days since outreach
    "election_year", # -> is_presidential, is_midterm
    "election_dow", # -> changed to categorical dummies
    "office_level", # -> office_level_cleaned
    "state", # -> state_usps (two letter code)
    "number_of_opponents", # -> number_of_opponents_num
    "incumbent", # -> incumbency_status (3 state variable)
    "open_seat", # -> incumbency_status (3 state variable)
    "seats_available", # -> number_avail_seats
    "days_between_outreach_and_election", # -> recency_weighted_days, recency_election_interaction
    "number_avail_seats", # -> competitiveness
    "number_of_opponents_num", # -> competitiveness
    "script" # -> sentiment features
    
    # Not included at the moment
    # Excluded for now because we were not given access to
    # all the values of this column in our dataset
    "most_common_outreach_type" # Only includes (text) types
    
]