import logging
import pandas as pd
from typing import Tuple
from core.preprocess import DataPreprocess
from core.preprocess import (
    DataImputeStrategy,
    FeatureEngineeringStrategy,
    DataTransformationStrategy,
    FeatureDroppStrategy,
    DataSplitStrategy,
)


def preprocess_data(df: pd.DataFrame) -> Tuple:
    """
    Component for cleaing data.
    Args:
        df: pandas.Dataframe
    """
    try:
        processed_df = DataPreprocess(df, DataImputeStrategy()).handle_data()
        processed_df = DataPreprocess(
            processed_df, FeatureEngineeringStrategy()
        ).handle_data()
        processed_df = DataPreprocess(
            processed_df, DataTransformationStrategy()
        ).handle_data()
        processed_df = DataPreprocess(
            processed_df, FeatureDroppStrategy()
        ).handle_data()
        X_train, X_test, y_train, y_test = DataPreprocess(
            processed_df, DataSplitStrategy()
        ).handle_data()
        logging.info("Data cleaning completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in preprocessing the data: {e}")
        raise e
