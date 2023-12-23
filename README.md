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
python3 -m venv mlops_env
source mlops_env/bin/activate
poetry install
pre-commit install
pre-commit run -a
python3 mlops_tools/train.py
python3 mlops_tools/infer.py
```

#### #3

To run **hw2** you should:

1. Specify `mlflow_uri` parameter in the
   [config](https://github.com/destitutiones/mlops-tools/blob/main/configs/config.yaml)
   (and have the server run).
2. Set `is_mlflow_logging` parameter `True`.
3. Run `./examples/hw2.sh` in the console.

**Note:** In case you want to infer a specific saved model, mind the
`inference_model` parameter.

### HW3 example run

1. First, load weights from dvc:

```bash
dvc pull
```

2. Next, run triton server:

```bash
cd nvidia_triton/
docker-compose up
```

3. Simultaniously run `client.py`:

```bash
python3 -m venv mlops_env
source mlops_env/bin/activate
poetry install
pre-commit install
pre-commit run -a
python3 nvidia_triton/client.py
```
