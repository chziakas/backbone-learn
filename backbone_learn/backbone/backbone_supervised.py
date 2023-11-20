from typing import List

import numpy as np

from ..utils.utils import Utils
from .backbone_base import BackboneBase


class BackboneSupervised(BackboneBase):
    """
    Implementation for supervised learning specific operations.
    """

    def preprocessing_backbone(self, X_selected: np.ndarray) -> np.ndarray:
        """
        Perform preprocessing specific to supervised learning during backbone construction.

        Args:
            X_selected (np.ndarray): The selected feature dataset after screen selection.

        Returns:
            np.ndarray: The preprocessed dataset, which is the same as input for supervised learning.
        """
        return X_selected

    def set_utilities(self, X: np.ndarray) -> np.ndarray:
        """
        Set utilities for supervised learning, typically one for each feature.

        Args:
            X (np.ndarray): The feature dataset.

        Returns:
            np.ndarray: An array of utilities, one for each feature.
        """
        return np.ones(X.shape[1])

    def utilize_variables(
        self, X_selected: np.ndarray, variables_exact_idx: List[int]
    ) -> np.ndarray:
        """
        Utilize selected variables in the dataset after processing the backbone sets.

        Args:
            X_selected (np.ndarray): The selected feature dataset after screen selection.
            variables_exact_idx (List[int]): List of indices for variables selected by the backbone.

        Returns:
            np.ndarray: Dataset with only the selected variables.
        """
        return X_selected[:, variables_exact_idx]

    def preprocessing_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the dataset before making predictions in supervised learning.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: Preprocessed dataset for prediction.
        """

        if self.screen_selector is not None:
            X = X[:, self.screen_selector.indices_keep]
        if self.heuristic_solver is not None:
            X = X[:, self.variables_exact_idx]
        return X

    def get_relevant_variables(self, feature_idx: List[int], threshold: float = None) -> List[int]:
        """
        Implements the retrieval of relevant variables for supervised learning.

        In supervised learning, this typically involves identifying variables or features
        that are relevant based on a certain threshold or criterion.

        Args:
            feature_idx (List[int]): List of feature indices to consider.
            threshold (float, optional): Threshold value for variable selection.

        Returns:
            List[int]: A list of indices representing the relevant variables.
        """
        # Get relevant variables from the heuristic solver
        rel_variables_local = self.heuristic_solver.get_relevant_variables(threshold)

        # Return the global indices of the relevant variables
        return [feature_idx[idx] for idx in rel_variables_local]

    def build_backbone_set(self, backbone_sets):
        """
        Merges a list of backbone sets into a single list, removes duplicates, and sorts the backbone list.

        Args:
            backbone_sets (list of list of int): The list of lists of backbone sets

        Returns:
            list: A backbone set sorted with unique elements.
        """

        return Utils.merge_lists_and_sort(backbone_sets)
