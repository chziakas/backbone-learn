import numpy as np

from .screen_selector_base import ScreenSelectorBase


class PearsonCorrelationSelector(ScreenSelectorBase):
    """
    Screen selector that uses Pearson correlation for calculating utilities.
    """

    def calculate_utilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate utilities based on Pearson correlation.
        """
        n_samples, n_features = X.shape
        utilities = np.zeros(n_features)

        y_mean = PearsonCorrelationSelector.compute_mean(y)
        y_std = PearsonCorrelationSelector.compute_std(y)

        for i in range(n_features):
            x_mean = PearsonCorrelationSelector.compute_mean(X[:, i])
            x_std = PearsonCorrelationSelector.compute_std(X[:, i])

            if x_std == 0 or y_std == 0:
                correlation = 0
            else:
                covariance = PearsonCorrelationSelector.compute_covariance(
                    X[:, i], y, x_mean, y_mean
                )
                correlation = covariance / (x_std * y_std)

            utilities[i] = np.abs(correlation)
        return utilities

    @staticmethod
    def compute_mean(array: np.ndarray) -> float:
        """
        Compute the mean of a numpy array.
        """
        return np.mean(array)

    @staticmethod
    def compute_std(array: np.ndarray) -> float:
        """
        Compute the standard deviation of a numpy array.
        """
        return np.std(array)

    @staticmethod
    def compute_covariance(x: np.ndarray, y: np.ndarray, x_mean: float, y_mean: float) -> float:
        """
        Compute the covariance between two numpy arrays.
        """
        return np.mean((x - x_mean) * (y - y_mean))
