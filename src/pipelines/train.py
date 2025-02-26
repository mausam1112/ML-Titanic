from components.ingest_data import ingest_data
from components.preprocess import preprocess_data
from components.model_train import train_model
from core.config import Configs
from core.utils import save_model


def train_pipeline(data_filepath: str):
    """
    Pipeline for training data.
    Args:
        data_filepath: str, path to data file
    """
    df = ingest_data(data_filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, name = train_model(X_train, y_train, Configs())
    save_model(model, name)
