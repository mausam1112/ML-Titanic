import logging
from pipelines.train import train_pipeline


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

data_path = "../data/titanic.csv"

if __name__ == "__main__":
    train_pipeline(data_path)
