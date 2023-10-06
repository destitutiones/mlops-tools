import os
import pickle
import sys


example_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(example_dir))
from mlops_tools.models import LinReg  # noqa: E402
from mlops_tools.utils import load_dataset, prepare_dataset  # noqa: E402


MODEL_NAME = "basic_lin_reg"


def main():
    reg = LinReg(MODEL_NAME)
    # Prepare & load diabetes datasets
    prepare_dataset(0.2)
    X_train, X_test, y_train, y_test = load_dataset()

    # Train model & save it to disk
    reg.fit(X_train, y_train)
    pickle.dump(reg, open(MODEL_NAME + ".sav", "wb"))


if __name__ == "__main__":
    main()
