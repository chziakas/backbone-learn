from abc import ABC, abstractmethod

import numpy as np


class VariableSelector(ABC):
    """
    Abstract base class for variable selectors.
    """

    @abstractmethod
    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Abstract method to select features.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target array.

        Returns:
            np.ndarray: The selected feature matrix.
        """
