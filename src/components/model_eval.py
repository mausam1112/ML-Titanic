import logging
import torch
from core.config import Configs
from core.model_eval import AccuracyEvaluator, F1Evaluator


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model on test data.
    Args:
        model: RegressorMixin
        X_test: pd.DataFrame
        y_test: pd.Series|np.ndarray
    Returns:
        float: accuracy
        float: fi-score
    """
    try:
        if Configs.model_name == "NN":
            y_pred = predict(model[0], X_test)
        else:
            y_pred = model.predict(X_test)

        accuracy = AccuracyEvaluator().evaluate(y_test, y_pred)
        f1 = F1Evaluator().evaluate(y_test, y_pred)
        return accuracy, f1
    except Exception as e:
        logging.error(f"Error calculating scores.. {e}")
        raise e


def predict(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(X_tensor).squeeze()
        predictions = (output.cpu().numpy() > 0.5).astype(int)
    return predictions
