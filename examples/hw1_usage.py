import os
import sys

import numpy as np
import pandas as pd
from sklearn import datasets


example_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(example_dir))
from mlops_tools.models import LinReg  # noqa: E402


def main():
    reg = LinReg("basic_lin_reg")
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    reg.fit(diabetes_X_train, diabetes_y_train)
    y_pred, metrics = reg.calculate_metrics(diabetes_X_test, diabetes_y_test)

    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
