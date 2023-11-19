from abc import abstractmethod

import numpy as np

from ..utils.utils import Utils
from .variable_selector import VariableSelector


class ScreenSelectorBase(VariableSelector):
    """
    Abstract base class for screen selectors.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize the ScreenSelectorBase with a default alpha value.

        Args:
            alpha (float): The proportion of features to retain after screening.
        """
        self.alpha = alpha
        self.utilities = None
        self.indices_keep = None

    @abstractmethod
    def calculate_utilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the utilities for each feature.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target array.

        Returns:
            np.ndarray: This method should return the utilities
        """

    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Selects features based on calculated utilities and alpha value.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target array.

        Returns:
            np.ndarray: The selected feature matrix.
        """
        n_samples, n_features = X.shape

        # Calculate utilities if not already done
        if self.utilities is None:
            self.utilities = self.calculate_utilities(X, y)

        # Determine the number of features to keep
        num_keep = int(self.alpha * n_features)

        # Select indices of the top utilities
        self.indices_keep = self.select_indices(self.utilities, num_keep)
        return X[:, self.indices_keep]

    @staticmethod
    def select_indices(utilities: np.ndarray, num_keep: int) -> np.ndarray:
        """
        Selects indices of the top utilities.

        Args:
            utilities (np.ndarray): Array of utilities for each feature.
            num_keep (int): Number of top features to keep.

        Returns:
            np.ndarray: Indices of the top utilities.
        """
        return Utils.find_idx_highest(utilities, num_keep)
