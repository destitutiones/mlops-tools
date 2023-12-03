import pickle
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd


mlops_tools_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.config import Params  # noqa: E402

from mlops_tools.models import CatBoostReg  # noqa: E402
from mlops_tools.utils import load_dataset  # noqa: E402


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    model_name = cfg["model"]["name"]
    is_mlflow_logging = cfg["common"]["is_mlflow_logging"]
    mlflow_models_path = cfg["common"]["mlflow_models_path"]
    inference_model = cfg["model"].get("inference_model")

    # Load datasets
    X_train, X_test, y_train, y_test = load_dataset(cfg["common"]["processed_data_path"])

    if is_mlflow_logging:
        # Use whether the mentioned in the config model to evaluate or the last saved one
        reg = CatBoostReg(
            name=model_name,
            model=mlflow.catboost.load_model(
                inference_model or f"{mlflow_models_path}{model_name}/"
            ),
        )
    else:
        # Load the model from disk
        reg = pickle.load(open(model_name + ".sav", "rb"))
    # Calculate metrics
    y_pred, metrics = reg.calculate_metrics(X_test, y_test)

    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
