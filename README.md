# mlops_tools

MIPT MLOps course project, autumn 2023

## Task

We're solving a regression task using
[California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).
Based on average number of rooms, bedrooms, household members, house age, etc.
we're predicting the median house value for California districts, expressed in
hundreds of dollars.

## Docs

You can read documentation
[here](https://github.com/destitutiones/mlops_tools/blob/main/docs/v0.0.1/index.md)

## Usage

### Poetry installation

[Docs](https://python-poetry.org/docs/#installation)

### Example run

#### #1

```
poetry install
poetry run python examples/hw1_usage.py
```

#### #2

```
poetry install
pre-commit install
pre-commit run -a
python mlops_tools/train.py
python mlops_tools/infer.py
```
