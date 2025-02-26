import logging
from pipelines.eval import eval_pipeline


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

data_path = "../data/titanic.csv"

if __name__ == "__main__":
    eval_pipeline(data_path, model_version=1)

    # Possible ways of execution
    # eval_pipeline(data_path, model_version='1')
    # eval_pipeline(data_path, model_version='v1')
