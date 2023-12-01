from dataclasses import dataclass
from typing import Any


@dataclass
class DiabetesData:
    name: str
    path: str
    test_size: float
    seed: int


@dataclass
class Model:
    name: str
    fit_intercept: bool
    n_jobs: int


@dataclass
class Params:
    data: Any
    model: Model
