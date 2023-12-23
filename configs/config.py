from dataclasses import dataclass
from typing import Any


@dataclass
class DiabetesData:
    name: str
    path: str
    test_size: float
    seed: int


@dataclass
class CaliforniaHousingData:
    name: str
    path: str
    test_size: float


@dataclass
class Model:
    name: str
    test_size: float
    inference_model: str
    onnx_model_name: str
    onnx_model_path: str
    params: Any


@dataclass
class CommonInfo:
    repo_url: str
    task: str
    raw_data_path: str
    processed_data_path: str
    is_mlflow_logging: bool
    mlflow_uri: str
    triton_path: str
    deviation_threshold: float


@dataclass
class Params:
    data: Any
    model: Model
    common: CommonInfo
