import logging
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod
from configs.configs import Configs
from core.data_loader import get_dataloaders
from models.model_titanic import TitanicClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        Args:
            X_train: Training data features.
            y_train: Training data labels.
        Returns:
            None
        """
        pass


class LogisticRegressionModel(Model):
    def train(self, X_train, y_train, *args, **kwargs):
        try:
            model_logistic = LogisticRegression(max_iter=1000)
            params = {
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": [
                    "lbfgs",
                    "liblinear",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ],
                "l1_ratio": [0, 0.25, 0.5, 0.75, 1.0],
            }
            grid_search = GridSearchCV(
                model_logistic, params, scoring="accuracy", n_jobs=-1
            )
            logging.info("Starting Logistic Model training.")
            grid_search.fit(X_train, y_train)
            logging.info("Logistic Model training completed.")
            logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best Score: {grid_search.best_score_}")
            logging.info(f"Best coef_: {grid_search.best_estimator_.coef_}")
            logging.info(f"Params: {grid_search.best_estimator_.get_params()}")
            logging.info(f"n_features_in_: {grid_search.feature_names_in_}")

            mlflow.set_tag("Model", "LogisticRegression")
            # mlflow.log_params(grid_search.best_estimator_.get_params())
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_score", grid_search.best_score_)

            return grid_search.best_estimator_
        except Exception as e:
            logging.error(f"Error in training logistic model with error {e}")
            raise e


class RandomForestModel(Model):
    def train(self, X_train, y_train, *args, **kwargs):
        try:
            model_rf = RandomForestClassifier()
            params = {
                "n_estimators": [50, 100, 200],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [3, 4, 5],
                "max_features": ["sqrt", "log2"],
            }
            grid_search = GridSearchCV(model_rf, params, scoring="accuracy", n_jobs=-1)

            logging.info("Starting RF Model training.")
            grid_search.fit(X_train, y_train)
            logging.info("RF Model training completed.")

            logging.info("Logistic Model training completed.")
            logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best Score: {grid_search.best_score_}")
            logging.info(f"Params: {grid_search.best_estimator_.get_params()}")
            logging.info(f"n_features_in_: {grid_search.feature_names_in_}")

            mlflow.set_tag("Model", "RandomForest")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_score", grid_search.best_score_)

            return grid_search.best_estimator_
        except Exception as e:
            logging.error(f"Error in training random forest model with error {e}")
            raise e


class TitanicNNModel(Model):
    def train(self, X_train, y_train, *args, **kwargs):
        try:
            model_nn = TitanicClassifier(X_train.shape[1])

            criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
            optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_nn.to(device)

            X_val = kwargs.get("X_val")
            y_val = kwargs.get("y_val")

            if X_val is None and y_val is None:
                raise ValueError(
                    "Validation data is required for training a neural network model"
                )
            train_loader = get_dataloaders(X_train, y_train)
            val_loader = get_dataloaders(X_val, y_val)

            # Early Stopping Parameters
            patience = 10  # Number of epochs to wait for improvement
            best_val_loss = float("inf")
            corresponding_train_loss = float("inf")
            corresponding_val_acc = 0
            corresponding_train_acc = 0
            counter = 0
            best_model = None

            for epoch in range(Configs.epochs):
                # Training
                model_nn.train()
                total_loss = 0
                correct_train = 0
                total_train = 0

                for batch in train_loader:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model_nn(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    # Accumulate training loss over the batch
                    total_loss += loss.item()
                    # accumulate training accuracy
                    preds = (outputs > 0.5).float()
                    correct_train += (preds == y_batch).sum().item()
                    total_train += y_batch.size(0)

                # Validation
                model_nn.eval()
                val_loss = 0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for batch in val_loader:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = model_nn(X_batch).squeeze()
                        loss = criterion(outputs, y_batch)

                        # Accumulate validation loss over the batch
                        val_loss += loss.item()

                        # Accumulate validation accuracy over the batch
                        preds = (outputs > 0.5).float()
                        correct_val += (preds == y_batch).sum().item()
                        total_val += y_batch.size(0)

                # Accuracy
                train_accuracy = correct_train / total_train
                val_accuracy = correct_val / total_val

                # Validation
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                logging.info(
                    " ".join(
                        [
                            f"Epoch {epoch+1}/{Configs.epochs},",
                            f"Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f},",
                            f"Val Acc: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}",
                        ]
                    )
                )

                # Early Stopping Check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    corresponding_train_loss = avg_train_loss
                    corresponding_train_acc = train_accuracy
                    corresponding_val_acc = val_accuracy
                    best_epoch = epoch + 1
                    counter = 0  # Reset patience counter
                    best_model = model_nn
                else:
                    counter += 1
                    if counter >= patience:
                        logging.info("Early stopping triggered. Stopping training.")
                        break

            mlflow.log_params(
                {
                    "patience": patience,
                    "max_epochs": Configs.epochs,
                    "batch_size": Configs.batch_size,
                    "best_epoch": best_epoch,
                    "final_epoch": epoch + 1,
                }
            )

            mlflow.log_metrics(
                {
                    "train_accuracy": corresponding_train_acc,
                    "val_accuracy": corresponding_val_acc,
                    "train_loss": corresponding_train_loss,
                    "val_loss": best_val_loss,
                }
            )
            return best_model if best_model else model_nn, Configs.model_name
        except Exception as e:
            logging.error(f"Error in training random forest model with error {e}")
            raise e
