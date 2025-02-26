import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
            f1 = f1_score(y_true, y_pred)
            return f1
        except Exception as e:
            logging.error(f"Error calculating f1 score: {e}")
            raise e


class PrecisionEvaluator(Evaluation):
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray):
        try:
            precision = precision_score(y_true, y_pred)
            return precision
        except Exception as e:
            logging.error(f"Error calculating f1 score: {e}")
            raise e


class RecallEvaluator(Evaluation):
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray):
        try:
            recall = recall_score(y_true, y_pred)
            return recall
        except Exception as e:
            logging.error(f"Error calculating f1 score: {e}")
            raise e
