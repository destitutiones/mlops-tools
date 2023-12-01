import pickle
import sys
from pathlib import Path

import hydra


mlops_tools_dir_path = Path(__file__).absolute().parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))


from configs.config import Params  # noqa: E402

from mlops_tools.models import LinReg  # noqa: E402
from mlops_tools.utils import load_dataset, prepare_dataset  # noqa: E402


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params):
    model_name = cfg["model"]["name"]
    reg = LinReg(model_name)
    # Prepare & load diabetes datasets
    prepare_dataset(0.2)
    X_train, X_test, y_train, y_test = load_dataset()

    # Train model & save it to disk
    reg.fit(X_train, y_train)
    pickle.dump(reg, open(model_name + ".sav", "wb"))


if __name__ == "__main__":
    main()
