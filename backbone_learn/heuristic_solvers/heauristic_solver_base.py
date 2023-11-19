from abc import ABC, abstractmethod

import numpy as np


class HeuristicSolverBase(ABC):
    """
    Abstract class for heuristic solvers.

    This class provides a framework for defining heuristic solvers that can fit models to data and identify relevant features.
    Derived classes need to implement the `fit` and `get_relevant_features` methods according to their specific heuristic approach.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits a model to the given data using a heuristic approach.

        This method should be implemented to solve a subproblem using the input data matrix X and the target vector y.
        It should fit a model based on a heuristic algorithm specific to the derived class.

        Args:
            X (np.ndarray): The input feature matrix.
            y (np.ndarray): The target vector.

        Returns:
            None: The method should fit the model to the data, with the results stored internally within the class instance.
        """

    @property
    def model(self):
        # Return the fitted model
        return self._model

    @abstractmethod
    def get_relevant_variables(self, **kwargs):
        """
        Identifies relevant variables with importance greater than a specified threshold.

        This method should be implemented to determine the most relevant variables based on the model fitted using the `fit` method.
        It should return the indices of variables that will be used for the exact solver
        """
