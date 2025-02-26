import joblib
import os
import torch


def save_model(model, name: str, base_dir="../saved_models"):
    directory = os.path.join(base_dir, name)
    os.makedirs(directory, exist_ok=True)

    if name == "NN":
        version = model_version(directory, ext=".pth")
        model_path = os.path.join(directory, f"v{version}.pth")
        torch.save(model, model_path)
    else:
        version = model_version(directory)
        model_path = os.path.join(directory, f"v{version + 1}.joblib")
        joblib.dump(model, model_path)


def model_version(directory: str, ext: str = ".joblib"):
    models = [f for f in os.listdir(directory) if f.endswith(ext)]
    if models:
        return max([int(m[1:].removesuffix(ext)) for m in models])
    else:
        return 0
