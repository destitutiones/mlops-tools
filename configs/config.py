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
    fit_intercept: bool
    n_jobs: int
    test_size: float


@dataclass
class CommonInfo:
    repo_url: str
    task: str
    raw_data_path: str
    processed_data_path: str
    mlflow_uri: str


@dataclass
class Params:
    data: Any
    model: Model
    common: CommonInfo
