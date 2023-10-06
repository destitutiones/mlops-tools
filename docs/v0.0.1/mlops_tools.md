# mlops_tools package

## Submodules

## mlops_tools.infer module

### mlops_tools.infer.main()

## mlops_tools.models module

### _class_ mlops_tools.models.LinReg(name: str)

Bases: `object`

Linear Regression model implementation. Source:
[https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)
(No one has told it should be at least a bit intellectual)

#### calculate_metrics(X_test, y_test)

Prediction and metrics calculation.

- **Parameters:**
  - **X_test** – Samples to predict values.
  - **y_test** – Target values to evaluate model quality.
- **Returns:** y_pred – model prediction, dict(‘model_name’, ‘mse’, ‘r2_score’)

#### fit(X_train, y_train)

Model training.

- **Parameters:**
  - **X_train** – Training data.
  - **y_train** – Target values.

#### plot_outputs(X_test, y_test, y_pred)

Sample and prediction rendering.

- **Parameters:**
  - **X_test** – Samples to predict values
  - **y_test** – Target values to evaluate model quality
  - **y_pred** – Values predicted by model

## mlops_tools.train module

### mlops_tools.train.main()

## mlops_tools.utils module

### mlops_tools.utils.load_dataset(path: str = './')

Reads train & test datasets from path.

- **Parameters:** **path** – path to saved train & test datasets in csv format
- **Returns:** Datasets: X_train, X_test, y_train, y_test

### mlops_tools.utils.prepare_dataset(test_size: float, path: str = './')

Saves train & test data from sklearn diabetes dataset by mentioned path in csv
format.

- **Parameters:**
  - **test_size** – should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split
  - **path** – path to save train & test data
- **Returns:**

## Module contents
