from load_data import load_training_data
from tuning import run_grid_search, run_grid_search_all_models
from sentiment import add_message_level_text_features
from feature_engineering import aggregate_message_level_data

full_df = load_training_data(spark).copy()

# message-level -> text features -> candidate-election level
full_df = add_message_level_text_features(full_df)
full_df = aggregate_message_level_data(full_df, training=True)

# Tune all models
all_results = run_grid_search_all_models(
    full_df,
    model_names=[
        #"logistic_regression",
        #"elastic_net_logistic",
        "random_forest",
        #"xgboost",
    ],
    scoring="roc_auc",
)

display(all_results["summary_df"])