import pickle
import sys
from pathlib import Path

import hydra


# import mlflow


mlops_tools_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.config import Params  # noqa: E402

from mlops_tools.models import CatBoostReg  # noqa: E402
from mlops_tools.utils import load_dataset, prepare_dataset  # noqa: E402


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    model_name = cfg["model"]["name"]
    seed = cfg["model"]["seed"]
    reg = CatBoostReg(model_name, seed)
    # Prepare & load diabetes datasets
    prepare_dataset(cfg)
    X_train, X_test, y_train, y_test = load_dataset(cfg["common"]["processed_data_path"])

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # exp_id = mlflow.set_experiment("check-train-py").experiment_id
    #
    # with mlflow.start_run():
    #     mlflow.log_metric("foo", 1)
    #     mlflow.log_metric("bar", 2)

    # Train model & save it to disk
    reg.fit(X_train, y_train)
    pickle.dump(reg, open(model_name + ".sav", "wb"))


if __name__ == "__main__":
    main()
