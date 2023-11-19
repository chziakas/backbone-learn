from typing import List

import numpy as np

from .subproblem_feature_selector import SubproblemFeatureSelector


class SubproblemConstructor:
    """
    Manages the construction of all subproblems.
    """

    def __init__(self, utilities: np.ndarray, beta: float, num_subproblems: int):
        """
        Initializes the SubproblemConstructor with given parameters.

        Args:
            utilities (np.ndarray): Array of feature utilities.
            beta (float): Proportion of screened features to use in each subproblem.
            num_subproblems (int): Number of subproblems to create.
        """
        self._utilities = utilities
        self._num_features = utilities.shape[0]
        self._beta = beta
        self._num_features_subproblem = int(np.ceil(beta * self._num_features))
        self._num_subproblems = num_subproblems

    @property
    def utilities(self) -> np.ndarray:
        """Returns the utilities of the features."""
        return self._utilities

    @property
    def num_features(self) -> int:
        """Returns the total number of features."""
        return self._num_features

    @property
    def beta(self) -> float:
        """Returns the beta proportion for subproblem feature selection."""
        return self._beta

    @property
    def num_features_subproblem(self) -> int:
        """Returns the number of features for each subproblem."""
        return self._num_features_subproblem

    @property
    def num_subproblems(self) -> int:
        """Returns the number of subproblems to create."""
        return self._num_subproblems

    def construct_subproblems(self) -> List[List[int]]:
        """
        Constructs and returns all subproblems.

        Returns:
            List[List[int]]: A list containing the indices of selected features for each subproblem.
        """
        subproblems = []
        selector = SubproblemFeatureSelector(self.utilities, self.num_features_subproblem)
        for _ in range(self.num_subproblems):
            subproblems.append(selector.select())

        return subproblems
