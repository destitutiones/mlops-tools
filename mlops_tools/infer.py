import pickle
import sys
from pathlib import Path

import hydra
import pandas as pd


mlops_tools_dir_path = Path(__file__).absolute().parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.config import Params  # noqa: E402

from mlops_tools.utils import load_dataset  # noqa: E402


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    # Load the model from disk
    reg = pickle.load(open(cfg["model"]["name"] + ".sav", "rb"))
    # Load diabetes datasets
    X_train, X_test, y_train, y_test = load_dataset()

    # Calculate metrics
    y_pred, metrics = reg.calculate_metrics(X_test, y_test)

    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
