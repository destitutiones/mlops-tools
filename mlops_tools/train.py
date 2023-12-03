import pickle
import sys
from pathlib import Path

import git
import hydra
import mlflow
import pandas as pd


mlops_tools_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.config import Params  # noqa: E402

from mlops_tools.models import CatBoostReg  # noqa: E402
from mlops_tools.utils import load_dataset, prepare_dataset  # noqa: E402


def mlflow_logging(model: CatBoostReg, sha: str) -> None:
    """
    Write model hyperparameters and metrics to mlflow.

    :param model: model training
    :param sha: current git commit id
    """
    mlflow.log_params(model.model.get_params())
    mlflow.log_param("git commit id", sha)

    evals_result = pd.DataFrame(model.model.get_evals_result()["learn"])
    for i, row in evals_result.iterrows():
        mlflow.log_metrics(row, i)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    repo_url = repo.remotes[0].config_reader.get("url")

    model_name = cfg["model"]["name"]
    model_params = cfg["model"]["params"]
    is_mlflow_logging = cfg["common"]["is_mlflow_logging"]

    reg = CatBoostReg(model_name, model_params)
    # Prepare & load diabetes datasets
    prepare_dataset(cfg, repo_url)
    X_train, X_test, y_train, y_test = load_dataset(cfg["common"]["processed_data_path"])

    if is_mlflow_logging:
        mlflow.set_tracking_uri(cfg["common"]["mlflow_uri"])
        mlflow.set_experiment(f"{model_name}_training")

        with mlflow.start_run():
            # Train model & save it to disk
            reg.fit(X_train, y_train)
            mlflow_logging(reg, sha)
    else:
        reg.fit(X_train, y_train)
    pickle.dump(reg, open(model_name + ".sav", "wb"))


if __name__ == "__main__":
    main()
