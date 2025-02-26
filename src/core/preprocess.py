import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        pass


class DataImputeStrategy(DataStrategy):
    """
    Imputing missing values
    """

    def handle_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        data["Age"] = data["Age"].fillna(
            data[["Pclass", "Sex", "Age"]]
            .groupby(["Pclass", "Sex"])["Age"]
            .transform("median")
        )

        # Filling the missing values with S
        data["Embarked"] = data["Embarked"].fillna("S")

        # fills the missing values with 'M' representing missing
        # Although deck M is in the plan, no data for deck M is present in dataset
        # We are using `M` for missing data
        data["Cabin"] = data["Cabin"].apply(lambda x: x if pd.notna(x) else "M")

        return data


class FeatureEngineeringStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        data["Deck"] = data["Cabin"].apply(lambda x: x[0])

        idx = data[data["Deck"] == "T"].index
        data.loc[idx, "Deck"] = "A"

        data["Deck"] = data["Deck"].replace(["A", "B", "C"], "ABC")

        # adding feature family size by adding number of siblings, spouse, parent and children
        data["Family_Size"] = data["SibSp"] + data["Parch"] + 1

        # grouping family size (number of members) into 4 categories (0-4)
        data["Family_Size"] = data["Family_Size"].apply(lambda x: self.group_family(x))

        data["Ticket_Frequency"] = data.groupby("Ticket")["Ticket"].transform("count")

        return data

    def group_family(self, family_size: int) -> int:
        match family_size:
            case 1:
                return 0  #'single'
            case 2 | 3 | 4:
                return 1  #'small'
            case 5 | 6 | 7:
                return 2  #'large'
            case _:
                return 3  # "v_large"


class DataTransformationStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        # Applying One-Hot Encoding to nominal features

        nominal_features = ["Pclass", "Sex", "Deck", "Embarked"]
        data = pd.get_dummies(data, columns=nominal_features, drop_first=True)

        return data


class FeatureDroppStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        drop_features = [
            "PassengerId",
            "Name",
            "SibSp",
            "Parch",
            "Ticket",
        ]
        data.drop(drop_features, axis=1, inplace=True)

        return data


class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, test_size=0.2) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Splits the data in train and test sets.

        Args:
            df: pandas.Dataframe
        """
        try:
            X = data.drop(["Survived"], axis=1)
            y = data["Survived"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e


class DataPreprocess:
    def __init__(self, data: pd.DataFrame | pd.Series, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        """
        Handling the data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e
