"""
Data loading utilities for training and scoring datasets.

This module contains helper functions for retrieving model input data from
Spark SQL and converting it into pandas DataFrames for downstream use in the
scikit-learn pipeline. It provides separate loaders for labeled training data
and scoring data, while keeping query execution logic centralized and reusable.
"""

from sql_query import TRAINING_QUERY, SCORING_QUERY, TRAINING_MESSAGE_QUERY, SCORING_MESSAGE_QUERY
import pandas as pd
import numpy as np


def load_training_data(spark):
    """
    Load the model training dataset from Spark SQL into a pandas DataFrame.

    This function executes the predefined `TRAINING_QUERY` against the active
    Spark session and converts the resulting Spark DataFrame to pandas for
    downstream modeling in scikit-learn.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session used to execute the SQL query.

    Returns
    -------
    pandas.DataFrame
        Training dataset returned by `TRAINING_QUERY`.

    Notes
    -----
    - Intended for labeled data used in model development and evaluation.
    - Assumes the query result is small enough to fit in memory, since
      `.toPandas()` collects all rows to the driver.
    """
    return spark.sql(TRAINING_MESSAGE_QUERY).toPandas()

def load_scoring_data(spark):
    """
    Load the scoring dataset from Spark SQL into a pandas DataFrame.

    This function executes the predefined `SCORING_QUERY` against the active
    Spark session and converts the resulting Spark DataFrame to pandas for
    downstream batch scoring or prediction generation.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session used to execute the SQL query.

    Returns
    -------
    pandas.DataFrame
        Scoring dataset returned by `SCORING_QUERY`.

    Notes
    -----
    - Intended for unlabeled or future data that will receive model scores.
    - Assumes the query result is small enough to fit in memory, since
      `.toPandas()` collects all rows to the driver.
    """
    return spark.sql(SCORING_MESSAGE_QUERY).toPandas()