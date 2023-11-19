import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .heauristic_solver_base import HeuristicSolverBase


class CARTDecisionTree(HeuristicSolverBase):
    """
    Implements a Classification And Regression Tree (CART) Decision Tree with cross-validation using AUC.
    This solver is a heuristic approach for fitting a decision tree model and identifying significant features.

    Attributes:
        _model (DecisionTreeClassifier): An instance of the sklearn DecisionTreeClassifier.
        _auc_score (float): The maximum AUC score obtained during cross-validation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the CARTDecisionTree with a DecisionTreeClassifier model.
        """
        self._model = DecisionTreeClassifier()
        self._auc_score = None

    @property
    def auc_score(self) -> float:
        """
        Returns the maximum AUC score obtained from cross-validation.

        Returns:
            float: The maximum AUC score.
        """
        return self._auc_score

    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> None:
        """
        Fits a CART Decision Tree model to the data using hyperparameter tuning with cross-validation and evaluates it using AUC.

        Args:
            X (np.ndarray): The input features as a NumPy array.
            y (np.ndarray): The target labels as a NumPy array.
            cv_folds (int): The number of folds to use for cross-validation.
        """
        # Define the parameter grid for hyperparameter tuning
        param_grid = {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4]}

        # Initialize GridSearchCV with the model and parameter grid
        grid_search = GridSearchCV(
            self._model, param_grid, cv=cv_folds, scoring="roc_auc", verbose=1
        )

        # Perform the grid search on the provided data
        grid_search.fit(X, y)

        # Update the model with the best found parameters
        self._model = grid_search.best_estimator_

        # Store the best AUC score
        self._auc_score = grid_search.best_score_

    def get_relevant_variables(self, threshold: float) -> np.ndarray:
        """
        Identifies features with importance greater than a specified threshold.

        Args:
            threshold (float): The threshold for determining feature relevance.

        Returns:
            np.ndarray: An array of indices of relevant features.
        """
        if self._model and hasattr(self._model, "feature_importances_"):
            significant_indices = np.where(self._model.feature_importances_ > threshold)[0]
            return significant_indices
        else:
            raise ValueError(
                "Model has not been fitted yet or does not support feature importances."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target labels for the given data.

        Args:
            X (np.ndarray): The input features as a NumPy array.

        Returns:
            np.ndarray: The predicted target labels.
        """
        if self._model is None:
            raise ValueError("The model has not been fitted yet.")

        return self._model.predict(X)
