import logging
import pandas as pd
import torch
from core.model_dev import LogisticRegressionModel, RandomForestModel, TitanicNNModel
from configs.configs import Configs
from sklearn.model_selection import train_test_split


def train_model(
    X_train: pd.DataFrame | torch.Tensor,
    y_train: pd.DataFrame | pd.Series | torch.Tensor,
    config: Configs,
):
    X_val, y_val = None, None
    try:
        if config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()
        elif config.model_name == "RandomForest":
            model = RandomForestModel()
        elif config.model_name == "NN":
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=0.9, random_state=42
            )
            X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)  # type: ignore
            X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)  # type: ignore
            y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)  # type: ignore
            y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
            model = TitanicNNModel()
        else:
            raise NotImplementedError(f"Model {config.model_name} is not implemented.")

        trained_model = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        return trained_model, config.model_name
    except Exception as e:
        logging.error(f"Error in training model. {e}")
        raise e
