from typing import Tuple

import pandas as pd
from dvc.api import DVCFileSystem
from sklearn.model_selection import train_test_split


def prepare_dataset(repo_url: str, test_size: float, path: str = "./") -> None:
    """
    Saves train & test data from sklearn diabetes dataset by mentioned path in csv format.

    :param repo_url: url to the current repository linked to dvc
    :param test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
    :param path: path to save train & test data
    :return:
    """

    assert 0 < test_size < 1, "Test share should be between 0.0 and 1.0"

    # Load the diabetes dataset
    fs = DVCFileSystem(repo_url, rev="main")
    fs.get("data", "data", recursive=True)

    diabetes_X = pd.read_csv("./data/diabetes_X.csv")
    diabetes_y = pd.read_csv("./data/diabetes_y.csv")
    # Split the data & targets into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes_X, diabetes_y, test_size=test_size, random_state=42
    )
    # Save datasets to csv
    X_train.to_csv(path + "X_train.csv")
    X_test.to_csv(path + "X_test.csv")
    y_train.to_csv(path + "y_train.csv")
    y_test.to_csv(path + "y_test.csv")


def load_dataset(
    path: str = "./",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads train & test datasets from path.

    :param path: path to saved train & test datasets in csv format
    :return: Datasets: X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(path + "X_train.csv")
    X_test = pd.read_csv(path + "X_test.csv")
    y_train = pd.read_csv(path + "y_train.csv")
    y_test = pd.read_csv(path + "y_test.csv")

    return X_train, X_test, y_train, y_test
