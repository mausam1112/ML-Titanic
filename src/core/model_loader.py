import joblib
import os
import torch
from core.utils import model_version, file_exists


def load_model(name: str, version: int | str = None, base_dir="../saved_models"):  # type: ignore
    directory = os.path.join(base_dir, name)
    assert os.path.exists(directory), f"{directory} doestn't exists."

    if not version:
        version_n = model_version(directory, ext=".pth" if name == "NN" else ".joblib")

        if version_n:
            version = f"v{version_n}"
        else:
            raise ValueError("No model file.")

        if version_n <= 0:
            raise ValueError("Invalid version number.")

    if isinstance(version, int) or (
        isinstance(version, str) and not version.startswith("v") and version.isnumeric()
    ):
        version = f"v{version}"

    if name == "NN":
        model_path = os.path.join(directory, f"{version}.pth")
        file_exists(model_path)
        return torch.load(model_path, weights_only=False)
    else:
        model_path = os.path.join(directory, f"{version}.joblib")
        file_exists(model_path)
        return joblib.load(model_path)
