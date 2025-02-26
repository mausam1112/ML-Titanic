import logging
from components.ingest_data import ingest_data
from components.preprocess import preprocess_data
from components.model_eval import evaluate_model
from core.model_loader import load_model
from configs.configs import Configs


def eval_pipeline(data_filepath: str, model_version: str | int = None):  # type: ignore
    """
    Pipeline for evaluation.
    Args:
        data_filepath: str, path to data file
    """
    df = ingest_data(data_filepath)
    _, X_test, _, y_test = preprocess_data(df)
    model = load_model(Configs.model_name, model_version)

    acc, f1 = evaluate_model(model, X_test, y_test)
    logging.info(f"Acc: {acc:.2f}, F1: {f1:.2f}")
