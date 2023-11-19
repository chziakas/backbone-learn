import numpy as np

from .screen_selector_base import ScreenSelectorBase


class LinearRegressionSelector(ScreenSelectorBase):
    """
    Screen selector that uses linear regression coefficients for calculating utilities.
    """

    def calculate_utilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate utilities based on the coefficients of a linear regression model.
        """
        # Add intercept term to X
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

        # Calculate coefficients using normal equation
        try:
            inv = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
            coefficients = np.dot(inv, np.dot(X_with_intercept.T, y))[1:]  # Exclude intercept
        except np.linalg.LinAlgError:
            # If X'X is not invertible, return zero utilities
            coefficients = np.zeros(X.shape[1])

        # Set utilities as the absolute value of coefficients
        utilities = np.abs(coefficients)
        return utilities
