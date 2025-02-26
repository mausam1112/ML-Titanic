import logging
from configs.constants import DATA_PATH
from pipelines.eval import eval_pipeline


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


if __name__ == "__main__":
    eval_pipeline(DATA_PATH, model_version=1)

    # Possible ways of execution
    # eval_pipeline(data_path, model_version='1')
    # eval_pipeline(data_path, model_version='v1')
