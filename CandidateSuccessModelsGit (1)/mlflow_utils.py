"""
Utility functions for MLflow experiment setup and model configuration.

This module centralizes a few small but important helpers used by the
training pipeline:

- ensuring an MLflow experiment exists and is active
- building a fully qualified Unity Catalog model name
- parsing command-line arguments for model selection
- validating that a selected model exists in MODEL_CONFIGS

These helpers keep the main training script cleaner and make model
selection/registration logic easier to reuse.
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

from config import (
    UC_CATALOG,
    UC_SCHEMA,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    MODEL_CONFIGS,
)


def ensure_experiment(path: str) -> str:
    """
    Ensure that an MLflow experiment exists at the given path and set it
    as the active experiment.
    """
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(path)

    if exp is None:
        experiment_id = client.create_experiment(path)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def get_registered_model_name(model_name: str) -> str:
    """
    Build the fully qualified Unity Catalog model name.
    """
    return f"{UC_CATALOG}.{UC_SCHEMA}.{model_name}_model"


def parse_args():
    """
    Parse command-line arguments and return the selected model name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=list(MODEL_CONFIGS.keys()),
    )
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    return args


def validate_model_config(model_name: str) -> dict:
    """
    Validate that the requested model exists in MODEL_CONFIGS and has the
    expected structure.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"MODEL_NAME='{model_name}' not found in MODEL_CONFIGS. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )

    model_spec = MODEL_CONFIGS[model_name]

    if "type" not in model_spec:
        raise ValueError(
            f"MODEL_CONFIGS['{model_name}'] must include a 'type' key."
        )

    if "params" not in model_spec:
        model_spec["params"] = {}

    return model_spec