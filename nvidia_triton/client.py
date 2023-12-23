import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


mlops_tools_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = mlops_tools_dir_path.parent
sys.path.append(str(mlops_tools_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.config import Params  # noqa: E402

from mlops_tools.models import CatBoostReg  # noqa: E402
from mlops_tools.utils import (  # noqa: E402; get_repo_params, load_onnx_model,
    calculate_metrics,
    load_dataset,
)


def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def make_predict(onnx_model_name: str, X_test: np.array) -> np.array:
    """
    Run saved in onnx-format model & make prediction.

    :param onnx_model_name: saved in onnx-format model name
    :param X_test: validation feature dataset
    :return: numpy array with predictions
    """
    triton_client = get_client()

    input = InferInput(name="features", shape=X_test.shape, datatype="FP32")
    input.set_data_from_numpy(X_test, binary_data=True)

    infer_output = InferRequestedOutput("predictions", binary_data=True)
    query_response = triton_client.infer(onnx_model_name, [input], outputs=[infer_output])

    predictions = query_response.as_numpy("predictions").squeeze(1)

    return predictions


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    model_name = cfg["model"]["onnx_model_name"]
    model_params = cfg["model"]["params"]

    # load_onnx_model(
    #     cfg["common"]["triton_weights_path"],
    #     cfg["common"]["triton_weights_target_path"],
    #     get_repo_params()[0],
    # )

    # triton part
    X_train, X_test, y_train, y_test = load_dataset(cfg["common"]["processed_data_path"])
    X_test = X_test.to_numpy(dtype=np.float32)

    predictions = make_predict(model_name, X_test)

    triton_res = calculate_metrics("triton_model", y_test, predictions)

    # standard part to compare
    reg = CatBoostReg("standard_model", model_params)
    reg.fit(X_train, y_train)
    _, standard_res = reg.calculate_metrics(X_test, y_test)

    print(pd.concat([triton_res, standard_res]))


if __name__ == "__main__":
    main()
