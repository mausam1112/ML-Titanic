import logging
import os
import pandas as pd

class IngestData:
    def __init__(self, data_filepath: str) -> None:
        """
        Args:
            data_filepath: string, path to data file 

        Returns:
            None
        """
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"{data_filepath} doesn't exists.")
        self.data_filepath = data_filepath

    def load_data(self):
        logging.info(f"Loading data from csv file at {self.data_filepath}.")
        return pd.read_csv(self.data_filepath)
    

def ingest_data(data_filepath: str) -> pd.DataFrame:
    """
    Ingesting the data from the file.

    Args:
        data_filepath: path to the file.
    Returns: 
        pandas Dataframe
    """
    try:
        in_data = IngestData(data_filepath)
        df = in_data.load_data()
        return df
    except Exception as e:
        logging.error(f"Error in loading data: {e}")
        raise e