from typing import List, Tuple

import numpy as np

from ..utils.utils import Utils
from .backbone_base import BackboneBase


class BackboneUnsupervised(BackboneBase):
    """
    Implementation for unsupervised learning specific operations.
    """

    def preprocessing_backbone(self, X_selected: np.ndarray) -> np.ndarray:
        """
        Perform preprocessing specific to unsupervised learning during backbone construction.
        This typically involves transposing the dataset.

        Args:
            X_selected (np.ndarray): The selected feature dataset after screen selection.

        Returns:
            np.ndarray: The transposed dataset for unsupervised learning.
        """
        return X_selected.transpose()

    def set_utilities(self, X: np.ndarray) -> np.ndarray:
        """
        Set utilities for unsupervised learning, typically one for each sample.

        Args:
            X (np.ndarray): The feature dataset.

        Returns:
            np.ndarray: An array of utilities, one for each sample.
        """
        return np.ones(X.shape[0])

    def utilize_variables(
        self, X_selected: np.ndarray, variables_exact_idx: List[int]
    ) -> np.ndarray:
        """
        Utilize selected variables in the dataset after processing the backbone sets in unsupervised learning.
        In unsupervised learning, the entire dataset is often used as is.

        Args:
            X_selected (np.ndarray): The selected feature dataset after screen selection.
            variables_exact_idx (List[int]): List of indices for variables selected by the backbone (unused in unsupervised).

        Returns:
            np.ndarray: Dataset with all features, as variable selection is typically not performed in unsupervised learning.
        """
        if self.heuristic_solver is not None:
            self.exact_solver.constraints = variables_exact_idx
        return X_selected

    def preprocessing_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the dataset before making predictions in unsupervised learning.
        Typically, the entire dataset is used as is.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: The original dataset, as preprocessing is typically not required for predictions in unsupervised learning.
        """
        return X

    def get_relevant_variables(
        self, feature_idx: List[int], threshold: float = None
    ) -> List[Tuple[int, int]]:
        """
        Implements the retrieval of relevant variables for unsupervised learning.

        In unsupervised learning, this method identifies pairs of variables that
        are considered relevant based on the learning model used.

        Args:
            feature_idx (List[int]): List of feature indices to consider.

        Returns:
            List[Tuple[int, int]]: A list of tuples, where each tuple contains a pair of indices
                                   representing relevant variable pairs.
        """
        rel_variables_local = self.heuristic_solver.get_relevant_variables()
        return rel_variables_local

    def build_backbone_set(self, backbone_sets) -> List:
        """
        Find tuples that are common to all backbone sets

        Args:
            backbone_sets (list of list of int): The list of lists of backbone sets

        Returns:
            list: A backbone set with the tuples
        """

        return Utils.find_common_tuples(backbone_sets)
