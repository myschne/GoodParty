"""
Ad hoc tuning driver for the Candidate Success modeling pipeline.

This script is an entry point for running grouped
hyperparameter tuning on one model or multiple models. It loads the training
data, applies the same text-feature and candidate-election aggregation steps
used in the main modeling workflow, and then calls the tuning helpers defined
in tuning.py.

This script supports:
- tuning a single selected model
- tuning multiple models in one run
- comparing best cross-validated results across model families

This file is intended for interactive experimentation and model
selection rather than the main production training workflow.
"""

from load_data import load_training_data
from tuning import run_grid_search, run_grid_search_all_models
from sentiment import add_message_level_text_features
from feature_engineering import aggregate_message_level_data

#################################
#============CONFIGS=============

# All models or one model?
tune_all_models = False

# Which models to tune?
one_model_name = "mixture_of_experts"

model_names=[
    "logistic_regression",
    "elastic_net_logistic",
    "random_forest",
    "mixture_of_experts",
    "xgboost",
]

#=================================

# Load in data
full_df = load_training_data(spark).copy()

# message-level -> text features -> candidate-election level
full_df = add_message_level_text_features(full_df)
full_df = aggregate_message_level_data(full_df, training=True)


if tune_all_models:
    # Tune all models
    all_results = run_grid_search_all_models(
        full_df,
        model_names=model_names,
        scoring="roc_auc",
    )

    display(all_results["summary_df"])

else:
    # Tune one model
    one_result = run_grid_search(
        full_df,
        model_name=one_model_name,
        scoring="roc_auc",
    )

    display(one_result)




