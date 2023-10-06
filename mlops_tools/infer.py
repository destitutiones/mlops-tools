import os
import pickle
import sys

import pandas as pd


example_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(example_dir))
from mlops_tools.utils import load_dataset  # noqa: E402


MODEL_NAME = "basic_lin_reg"


def main():
    # Load the model from disk
    reg = pickle.load(open(MODEL_NAME + ".sav", "rb"))
    # Load diabetes datasets
    X_train, X_test, y_train, y_test = load_dataset()

    # Calculate metrics
    y_pred, metrics = reg.calculate_metrics(X_test, y_test)

    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
