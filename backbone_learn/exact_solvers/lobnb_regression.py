import numpy as np

from .exact_solver_base import ExactSolverBase
from .lobnb_regression_model import L0BnBRegressionModel


class L0BnBRegression(ExactSolverBase):
    """
    Implements a regression solver using the L0BnB method for feature selection.

    This class is designed to provide an easy-to-use interface for the L0BnB regression model, allowing for fitting and predictions on datasets.

    Attributes:
        model (L0BnBRegressionModel): An instance of the L0BnBRegressionModel class.
    """

    def __init__(self, lambda_2: float = 0.01, max_nonzeros: int = 10, time_limit: int = 1000):
        """
        Initializes the L0BnBRegression with specified parameters for the L0BnB optimization process.

        Args:
            lambda_2 (float): Regularization parameter lambda_2 for the L0BnB model.
            max_nonzeros (int): Maximum number of non-zero coefficients allowed in the model.
            time_limit (int): Time limit for the optimization process.
        """
        self._model = L0BnBRegressionModel(lambda_2, max_nonzeros, time_limit)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the L0BnB regression model to the provided data.

        Args:
            X (np.ndarray): The feature matrix for the regression model.
            y (np.ndarray): The target values for the regression model.
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions using the fitted L0BnB regression model.

        This method delegates the prediction task to the `predict` method of the `L0BnBRegressionModel` instance. It requires the model to be already fitted.

        Args:
            X (np.ndarray): The feature matrix for which predictions are to be made.

        Returns:
            np.ndarray: Predicted values based on the fitted model.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self._model:
            return self._model.predict(X)
        else:
            raise ValueError(
                "The model has not been fitted yet. Please fit the model before making predictions."
            )
