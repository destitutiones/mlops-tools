from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


class CatBoostReg:
    """
    CatBoost Regressor model implementation.
    """

    def __init__(self, name: str, params: dict) -> None:
        """
        :param name: model name
        :param params: model parameters
        """
        self.name = name
        self.model = CatBoostRegressor(**params)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Model training.

        :param X_train: Training data.
        :param y_train: Target valuese.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def calculate_metrics(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> Tuple[list[float], dict]:
        """
        Prediction and metrics calculation.

        :param X_test: Samples to predict values, pd.DataFrame.
        :param y_test: Target values to evaluate model quality, pd.DataFrame.
        :return:
            y_pred – model prediction,
            dict('model_name', 'mse', 'r2_score')
        """
        assert self.is_fitted, "Train your model using `fit` method"
        y_pred = self.model.predict(X_test)
        mse_val = mean_squared_error(y_test, y_pred)
        r2_score_val = r2_score(y_test, y_pred)
        metrics = pd.DataFrame(
            {
                "model_name": [self.name],
                "mse": [mse_val],
                "r2_score": [r2_score_val],
            }
        )
        return y_pred, metrics

    def plot_outputs(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame
    ) -> None:
        """
        Sample and prediction rendering.

        :param X_test: Samples to predict values
        :param y_test: Target values to evaluate model quality
        :param y_pred: Values predicted by model
        """
        plt.scatter(X_test, y_test, color="black")
        plt.plot(X_test, y_pred, color="blue", linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()
