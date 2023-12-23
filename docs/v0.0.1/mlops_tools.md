# mlops_tools package

## Submodules

## mlops_tools.infer module

### mlops_tools.infer.main(cfg: Params)

## mlops_tools.models module

### _class_ mlops_tools.models.CatBoostReg(name: str, params: dict | None = None, model: CatBoostRegressor | None = None)

Bases: `object`

CatBoost Regressor model implementation.

#### calculate_metrics(X_test: DataFrame, y_test: DataFrame)

Prediction and metrics calculation.

- **Parameters:**
  - **X_test** – Samples to predict values, pd.DataFrame.
  - **y_test** – Target values to evaluate model quality, pd.DataFrame.
- **Returns:** y_pred – model prediction, dict(‘model_name’, ‘mse’, ‘r2_score’)

#### fit(X_train: DataFrame, y_train: DataFrame)

Model training.

- **Parameters:**
  - **X_train** – Training data.
  - **y_train** – Target valuese.

#### plot_outputs(X_test: DataFrame, y_test: DataFrame, y_pred: DataFrame)

Sample and prediction rendering.

- **Parameters:**
  - **X_test** – Samples to predict values
  - **y_test** – Target values to evaluate model quality
  - **y_pred** – Values predicted by model

## mlops_tools.train module

### mlops_tools.train.main(cfg: Params)

### mlops_tools.train.mlflow_logging(model: [CatBoostReg](#mlops_tools.models.CatBoostReg), sha: str)

Write model hyperparameters and metrics to mlflow.

- **Parameters:**
  - **model** – model training
  - **sha** – current git commit id

## mlops_tools.utils module

### mlops_tools.utils.get_repo_params()

Get current repo url & current sha. :return: repo_url, sha

### mlops_tools.utils.load_dataset(processed_data_path: str)

Reads train & test datasets from path.

- **Parameters:** **processed_data_path** – path to saved train & test datasets
  in csv format
- **Returns:** Datasets: X_train, X_test, y_train, y_test

### mlops_tools.utils.prepare_dataset(cfg: Params, repo_url: str)

Saves train & test data from sklearn diabetes dataset by mentioned path in csv
format.

- **Parameters:**
  - **cfg** – Params config from train run
  - **repo_url** – url to the current repository linked to dvc
- **Returns:**

## Module contents
