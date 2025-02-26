import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass


class AccuracyEvaluator(Evaluation):
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray):
        try:
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            raise e


class F1Evaluator(Evaluation):
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray):
        try:
            accuracy = f1_score(y_true, y_pred)
            return accuracy
        except Exception as e:
            logging.error(f"Error calculating f1 score: {e}")
            raise e
