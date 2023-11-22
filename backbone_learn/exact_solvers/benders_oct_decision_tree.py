import numpy as np
from odtlearn.flow_oct import BendersOCT
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from .exact_solver_base import ExactSolverBase


class BendersOCTDecisionTree(ExactSolverBase):
    """
    Implements the BendersOCT model for Optimal Classification Trees.

    Attributes:
        model (BendersOCT): The BendersOCT model.
        accuracy_score (float): The accuracy score of the trained model.
        est_X (KBinsDiscretizer): The KBinsDiscretizer instance for discretizing features.
        enc (OneHotEncoder): The OneHotEncoder instance for encoding categorical variables.
    """

    def __init__(
        self,
        depth=3,
        time_limit=1000,
        _lambda=0.5,
        num_threads=None,
        obj_mode="acc",
        n_bins=2,
        is_data_fit=False,
    ):
        """
        Initializes the BendersOCTDecisionTree with default or specified values.

        Args:
            depth (int): Maximum depth of the tree.
            time_limit (int): Time limit for the optimization process.
            _lambda (float): Regularization parameter.
            num_threads (int or None): Number of threads for parallel processing.
            obj_mode (str): Objective mode, e.g., 'acc' for accuracy.
            n_bins (int): Number of bins for KBinsDiscretizer.
            is_data_fit (bool): Whether data are in the format required for OCT
        """
        super().__init__()
        self._model = BendersOCT(
            solver="cbc",
            depth=depth,
            time_limit=time_limit,
            _lambda=_lambda,
            num_threads=num_threads,
            obj_mode=obj_mode,
        )
        self.accuracy_score = None
        self.est_X = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        self.enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
        self.is_data_fit = is_data_fit

    @property
    def model(self):
        """
        Returns the BendersOCT model instance.
        """
        return self._model

    def preprocess_features(self, X):
        """
        Transforms the features using already fitted KBinsDiscretizer and OneHotEncoder.

        Args:
            X (np.ndarray or pd.DataFrame): The input features (test data).

        Returns:
            np.ndarray: Transformed features.
        """
        X_bin = self.est_X.transform(X)
        return self.enc.transform(X_bin).toarray()

    def fit_preprocessors(self, X_train):
        """
        Fits preprocessors to the training data.

        Args:
            X_train (np.ndarray or pd.DataFrame): The training features.
        """
        self.est_X.fit(X_train)
        X_bin = self.est_X.transform(X_train)
        self.enc.fit(X_bin)

    def fit(self, X, y) -> None:
        """
        Fits the BendersOCT model to the preprocessed data.

        Args:
            X (np.ndarray or pd.DataFrame): The input features.
            y (np.ndarray or pd.Series): The target variable.
        """
        if self.is_data_fit:
            X_preprocessed = X
        else:
            self.fit_preprocessors(X)
            X_preprocessed = self.preprocess_features(X)
        self._model.fit(X_preprocessed, y)

    def predict(self, X) -> np.ndarray:
        """
        Predicts using the fitted BendersOCT model.

        Args:
            X (np.ndarray or pd.DataFrame): The input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.is_data_fit:
            X_preprocessed = X
        else:
            X_preprocessed = self.preprocess_features(X)
        return self._model.predict(X_preprocessed)
