"""
Scoring script for applying a registered MLflow model to current candidate data.

This script:
1. Parses the selected model name from the command line
2. Builds the registered model URI from Unity Catalog metadata
3. Loads current scoring data from Spark
4. Loads the trained model from MLflow Model Registry
5. Generates candidate-level predictions
6. Writes scored predictions back to a Unity Catalog table
7. Returns both pandas and Spark outputs for inspection

This file is intended to be the scoring/inference entry point, separate
from model training. It assumes a trained model has already been registered
in MLflow and assigned the alias specified in config.py.
"""

import argparse
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F

from load_data import load_scoring_data
from scoring import score_candidates
from config import (
    MODEL_NAME as DEFAULT_MODEL_NAME,
    UC_CATALOG,
    UC_SCHEMA,
    MODEL_ALIAS,
    MODEL_CONFIGS,
)

from sentiment import add_message_level_text_features
from feature_engineering import aggregate_message_level_data


# =========================================================
# Configuration
# =========================================================

# Use Databricks Unity Catalog as the MLflow model registry backend.
mlflow.set_registry_uri("databricks-uc")


def parse_args():
    """
    Parse command-line arguments for scoring.

    Supported arguments
    -------------------
    --model_name : str
        Name of the model configuration to use for scoring. Must be one of
        the keys in MODEL_CONFIGS. Defaults to DEFAULT_MODEL_NAME.

    Returns
    -------
    argparse.Namespace
        Parsed arguments object containing the selected model name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name to score with",
    )
    return parser.parse_args()


def get_registered_model_name(model_name: str) -> str:
    """
    Build the fully qualified registered model name in Unity Catalog.

    The scoring workflow assumes each trained model is registered under:
        <catalog>.<schema>.<model_name>_model

    Parameters
    ----------
    model_name : str
        Short project model name.

    Returns
    -------
    str
        Fully qualified registered model name.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_model"


def get_model_uri(model_name: str) -> str:
    """
    Build the MLflow model URI for the selected registered model alias.

    This uses the configured MODEL_ALIAS so scoring always points to a stable,
    registry-managed version such as 'Champion', 'Production', or another alias.

    Parameters
    ----------
    model_name : str
        Short project model name.

    Returns
    -------
    str
        MLflow model URI in the format:
        models:/<registered_model_name>@<MODEL_ALIAS>
    """
    registered_model_name = get_registered_model_name(model_name)
    return f"models:/{registered_model_name}@{MODEL_ALIAS}"


def get_output_table_name(model_name: str) -> str:
    """
    Build the fully qualified Unity Catalog output table name for predictions.

    Parameters
    ----------
    model_name : str
        Short project model name.

    Returns
    -------
    str
        Fully qualified table name where scored predictions will be written.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}"


# =========================================================
# Scoring workflow
# =========================================================

def main(spark, model_name):
    """
    Run the end-to-end scoring workflow for a selected model.

    Workflow steps
    --------------
    1. Build the registered MLflow model URI and output table name
    2. Load current scoring data from Spark
    3. Load the trained model from MLflow Model Registry
    4. Score candidates using the project scoring function
    5. Convert scored predictions back to Spark
    6. Add scoring metadata columns
    7. Overwrite the target Unity Catalog table with fresh predictions

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    model_name : str
        Name of the model to score with.

    Returns
    -------
    dict
        Dictionary containing:
        - score_df : pandas.DataFrame
            Raw scoring input data
        - predictions_pd : pandas.DataFrame
            Candidate-level predictions in pandas format
        - predictions_spark : pyspark.sql.DataFrame
            Predictions converted to Spark and enriched with metadata
        - model_uri : str
            MLflow model URI used for scoring
        - output_table : str
            Unity Catalog table written with predictions
    """
    model_uri = get_model_uri(model_name)
    output_table = get_output_table_name(model_name)

    # 1) Load current scoring data from Spark
    score_df = load_scoring_data(spark)
    print(f"Loaded {len(score_df)} raw rows for scoring.")

    score_df = add_message_level_text_features(score_df)
    score_df = aggregate_message_level_data(score_df, training=False)
    print(f"Loaded {len(score_df)} aggregated rows for scoring.")

    # 2) Load trained model from MLflow
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from: {model_uri}")

    # 3) Score candidates
    predictions_pd = score_candidates(model, score_df)
    print(f"Generated {len(predictions_pd)} predictions.")

    # 4) Convert predictions back to Spark and attach metadata
    predictions_spark = (
        spark.createDataFrame(predictions_pd)
        .withColumn("scored_at", F.current_timestamp())
        .withColumn("model_uri", F.lit(model_uri))
    )

    # 5) Write predictions table to Unity Catalog
    predictions_spark.write.mode("overwrite").saveAsTable(output_table)
    print(f"Wrote predictions table to: {output_table}")

    return {
        "score_df": score_df,
        "predictions_pd": predictions_pd,
        "predictions_spark": predictions_spark,
        "model_uri": model_uri,
        "output_table": output_table,
    }


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    args = parse_args()
    outputs = main(spark, model_name=args.model_name)

    print("\nScoring complete.")
    print("Model URI:", outputs["model_uri"])
    print("\nSample predictions:")
    print(outputs["predictions_pd"].head())

    display(outputs["predictions_spark"])