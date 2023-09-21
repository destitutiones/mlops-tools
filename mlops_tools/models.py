from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


class LinReg:
    '''
    Linear Regression model implementation.
    Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    (No one has told it should be at least a bit intellectual)
    '''

    def __init__(self, name: str):
        '''
        :param name: model name
        '''
        self.model = linear_model.LinearRegression()
        self.name = name
        self.is_fitted = False

    def fit(self, X_train, y_train) -> None:
        '''
        Model training.

        :param X_train: Training data.
        :param y_train: Target values.
        '''
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def calculate_metrics(self, X_test, y_test) -> Tuple[list[float], dict]:
        '''
        Prediction and metrics calculation.

        :param X_test: Samples to predict values.
        :param y_test: Target values to evaluate model quality.
        :return:
            y_pred – model prediction,
            dict('model_name', 'mse', 'r2_score')
        '''
        assert self.is_fitted, 'Train your model using `fit` method'
        y_pred = self.model.predict(X_test)
        mse_val = mean_squared_error(y_test, y_pred)
        r2_score_val = r2_score(y_test, y_pred)
        metrics = pd.DataFrame({
            'model_name': [self.name],
            'mse': [mse_val],
            'r2_score': [r2_score_val]
        })
        return y_pred, metrics

    def plot_outputs(self, X_test, y_test, y_pred) -> None:
        '''
        Sample and prediction rendering.

        :param X_test: Samples to predict values
        :param y_test: Target values to evaluate model quality
        :param y_pred: Values predicted by model
        '''
        plt.scatter(X_test, y_test, color="black")
        plt.plot(X_test, y_pred, color="blue", linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()
