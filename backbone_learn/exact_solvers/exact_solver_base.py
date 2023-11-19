from abc import ABC, abstractmethod

import numpy as np


class ExactSolverBase(ABC):
    """
    Abstract class for solving subproblems in various contexts.

    This class provides a framework for defining solvers that can fit models to data and make predictions.
    Derived classes need to implement the `fit` and `predict` methods according to the specifics of the solver.
    """

    @property
    def model(self):
        """
        This property should be implemented by subclasses to return the model instance used in the exact approach.
        """
        return self._model

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits a model to the given data.

        This method should be implemented to solve a subproblem using the input data matrix X and the target vector y.
        It should fit a model based on the specific algorithm implemented in the derived class.

        Args:
            X (np.ndarray): The input feature matrix.
            y (np.ndarray): The target vector.

        Returns:
            None: The method should fit the model to the data, with the results stored internally within the class instance.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        This method should be implemented to provide predictions based on the model fitted using the `fit` method.
        It should process the input feature matrix X and return predictions.

        Args:
            X (np.ndarray): The input feature matrix for which predictions are to be made.

        Returns:
            np.ndarray: An array of predictions made by the model.
        """
