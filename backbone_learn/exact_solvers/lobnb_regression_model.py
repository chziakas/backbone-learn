import numpy as np
from l0bnb import fit_path


class L0BnBRegressionModel:
    """
    A wrapper class for the L0BnB regression model, facilitating model fitting and predictions.

    This class serves as a convenient interface for the L0BnB feature selection method, specifically tailored for regression tasks. It stores the regression coefficients and intercept after fitting the model to the provided data.

    Attributes:
        coefficients (np.ndarray or None): Coefficients of the regression model. None before the model is fitted.
        intercept (float or None): Intercept of the regression model. None before the model is fitted.
        lambda_2 (float): Regularization parameter lambda_2 in the L0BnB model, controlling the trade-off between the model's complexity and fit.
        max_nonzeros (int): Maximum number of non-zero coefficients the model is allowed to have, enforcing sparsity.
    """

    def __init__(self, lambda_2: float, max_nonzeros: int, time_limit: int):
        """
        Initializes the L0BnBRegressionModel with specified parameters for L0BnB optimization.

        Args:
            lambda_2 (float): Regularization parameter for the L0BnB optimization process.
            max_nonzeros (int): Constraint on the maximum number of non-zero coefficients in the model.
            time_limit (int): Time limit for the optimization process.
        """
        self.coefficients = None
        self.intercept = None
        self.lambda_2 = lambda_2
        self.max_nonzeros = max_nonzeros
        self.time_limit = time_limit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fits the L0BnB regression model to the given dataset.

        The method uses the L0BnB algorithm to find a sparse set of coefficients that best fit the data.

        Args:
            X (np.ndarray): The feature matrix for the regression model.
            y (np.ndarray): The target values for the regression model.

        """

        solutions = fit_path(
            X, y, lambda_2=self.lambda_2, max_nonzeros=self.max_nonzeros, time_limit=self.time_limit
        )
        if solutions:
            selected_model = solutions[-1]
            self.coefficients = selected_model["B"]
            self.intercept = selected_model["B0"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the given input features using the fitted L0BnB regression model.

        Args:
            X (np.ndarray): The feature matrix for making predictions.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            ValueError: If the model is not fitted (coefficients are None).
        """
        if self.coefficients is not None:
            return np.dot(X, self.coefficients) + self.intercept
        else:
            raise ValueError("Model has not been fitted yet. Coefficients are not set.")
