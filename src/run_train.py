import logging
from constants.constants import DATA_PATH
from pipelines.train import train_pipeline


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


if __name__ == "__main__":
    train_pipeline(DATA_PATH)
