import numpy as np
from sklearn.linear_model import LassoCV

from .heauristic_solver_base import HeuristicSolverBase


class LassoRegression(HeuristicSolverBase):
    """
    Implements Lasso regression for feature selection using cross-validation.

    This class uses Lasso (Least Absolute Shrinkage and Selection Operator) regression,
    which is a type of linear regression that uses shrinkage. Shrinkage is where data values
    are shrunk towards a central point, like the mean. The lasso procedure encourages simple,
    sparse models (i.e. models with fewer parameters).

    Attributes:
        _model (LassoCV): The LassoCV regression model.
        _mse_score (float): The mean squared error score of the trained model.
    """

    def __init__(self, random_state=0, **kwargs):
        """
        Initializes the LassoRegression with specified cross-validation folds and random state.

        Args:
            cv_folds (int): The number of cross-validation folds to use. Default is 5.
            random_state (int): The seed used by the random number generator. Default is 0.
        """
        self._model = LassoCV()
        self._mse_score = None

    @property
    def mse_score(self) -> float:
        """
        Returns the mean squared error score of the trained model.

        Returns:
            float: The mean squared error score.
        """
        return self._mse_score

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alphas=None,
        max_iter=1000,
        tol=0.0001,
        selection="cyclic",
        cv_folds=5,
        random_state=0,
    ) -> None:
        """
        Fits a sparse regression model to the data using LassoCV.

        Args:
            X (np.ndarray): The input feature matrix.
            y (np.ndarray): The target variable.
            alphas (array-like, optional): List of alphas where to compute the models. If None alphas are set automatically.
            max_iter (int): The maximum number of iterations.
            tol (float): The tolerance for the optimization.
            selection (str): If set to 'random', a random coefficient is updated every iteration.
        """
        self._model.set_params(
            cv=cv_folds,
            random_state=random_state,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
        )
        self._model.fit(X, y)  # Fit the _model on the dataset

    def get_relevant_variables(self, threshold: float) -> np.ndarray:
        """
        Identifies features with coefficients greater than a specified threshold.

        Args:
            threshold (float): The threshold for determining feature relevance.

        Returns:
            np.ndarray: Indices of features whose coefficients are above the threshold.
        """
        if self._model:
            significant_indices = np.where(np.abs(self._model.coef_) > threshold)[0]
            return significant_indices
        else:
            raise ValueError("Model has not been fitted yet.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given data using the trained Lasso model.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: The predicted target values.
        """
        if self._model is None:
            raise ValueError("The model has not been fitted yet.")

        return self._model.predict(X)

    def keep_top_features(self, n_non_zeros: int) -> None:
        """
        Retain only the top 'n_non_zeros' features in the Lasso model.

        Args:
        n_non_zeros (int): Number of features to retain.
        """
        if self._model is None:
            raise ValueError("Model must be fitted before adjusting features.")

        # Get the absolute values of the coefficients
        coef_magnitude = np.abs(self._model.coef_)

        # Find the threshold for the top 'n_non_zeros' coefficients
        threshold = (
            np.sort(coef_magnitude)[-n_non_zeros] if n_non_zeros < len(coef_magnitude) else 0
        )

        # Zero out coefficients that are below the threshold
        self._model.coef_[coef_magnitude < threshold] = 0
