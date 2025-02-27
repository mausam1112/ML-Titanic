import logging
import mlflow
from components.ingest_data import ingest_data
from components.preprocess import preprocess_data
from components.model_eval import evaluate_model
from components.model_train import train_model
from configs.configs import Configs
from core.utils import save_model


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


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

    acc, f1, precision, recall = evaluate_model(model, X_test, y_test)
    logging.info(f"Acc: {acc:.2f}, F1: {f1:.2f}, {precision:.2f}, F1: {recall:.2f}")
    mlflow.log_metrics(
        {"accuracy": acc, "f1-score": f1, "precision": precision, "recall": recall}  # type: ignore
    )
