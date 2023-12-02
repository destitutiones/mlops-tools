from typing import Tuple

import pandas as pd
from dvc.api import DVCFileSystem
from sklearn.model_selection import train_test_split


def prepare_dataset(
    cfg: dict,
) -> None:  # repo_url: str, test_size: float, path: str = "./") -> None:
    """
    Saves train & test data from sklearn diabetes dataset by mentioned path in csv format.

    :param repo_url: url to the current repository linked to dvc
    :param test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
    :param path: path to save train & test data
    :return:
    """

    repo_url = cfg["common"]["repo_url"]
    test_size = cfg["model"]["test_size"]
    raw_data_path = cfg["common"]["raw_data_path"]
    raw_data_path_loaded = raw_data_path + "loaded/"
    processed_data_path = cfg["common"]["processed_data_path"]
    data_name = cfg["common"]["data"]

    assert 0 < test_size < 1, "Test share should be between 0.0 and 1.0"

    # Load the diabetes dataset
    fs = DVCFileSystem(repo_url, rev="main")
    fs.get(raw_data_path, raw_data_path_loaded, recursive=True)

    data_X = pd.read_csv(f"{raw_data_path_loaded}{data_name}_X.csv", index_col=0)
    data_y = pd.read_csv(f"{raw_data_path_loaded}{data_name}_y.csv", index_col=0)
    # Split the data & targets into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=test_size, random_state=42
    )
    # Save datasets to csv
    X_train.to_csv(processed_data_path + "X_train.csv", index=False)
    X_test.to_csv(processed_data_path + "X_test.csv", index=False)
    y_train.to_csv(processed_data_path + "y_train.csv", index=False)
    y_test.to_csv(processed_data_path + "y_test.csv", index=False)


def load_dataset(
    processed_data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads train & test datasets from path.

    :param processed_data_path: path to saved train & test datasets in csv format
    :return: Datasets: X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(processed_data_path + "X_train.csv")
    X_test = pd.read_csv(processed_data_path + "X_test.csv")
    y_train = pd.read_csv(processed_data_path + "y_train.csv")
    y_test = pd.read_csv(processed_data_path + "y_test.csv")

    return X_train, X_test, y_train, y_test
