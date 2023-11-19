from typing import List

import numpy as np

from ..screen_selectors.variable_selector import VariableSelector


class SubproblemFeatureSelector(VariableSelector):
    """
    Selects features for a single subproblem based on utilities.
    """

    def __init__(self, utilities: np.ndarray, num_features_to_select: int):
        """
        Initializes the SubproblemFeatureSelector with given parameters.

        Args:
            utilities (np.ndarray): Array of feature utilities.
            num_features_to_select (int): Number of features to select for the subproblem.
        """
        self._utilities = utilities
        self._num_features_to_select = num_features_to_select
        self._probability_distribution = self.compute_probability_distribution(utilities)

    @property
    def utilities(self) -> np.ndarray:
        """Returns the utilities of the features."""
        return self._utilities

    @property
    def probability_distribution(self) -> np.ndarray:
        """Returns the probability distribution for feature selection."""
        return self._probability_distribution

    @property
    def num_features_to_select(self) -> int:
        """Returns the number of features to select for a subproblem."""
        return self._num_features_to_select

    @staticmethod
    def compute_probability_distribution(utilities: np.ndarray) -> np.ndarray:
        """
        Computes the probability distribution for selecting features.

        Args:
            utilities (np.ndarray): Array of feature utilities.

        Returns:
            np.ndarray: Normalized probability distribution based on utilities.
        """
        normalized_utilities = utilities / np.max(utilities)
        exp_utilities = np.exp(normalized_utilities + 1)
        probability_distribution = exp_utilities / exp_utilities.sum()
        return probability_distribution

    def select(self) -> List[int]:
        """
        Samples a subset of features based on computed probability distribution.

        Returns:
            List[int]: Indices of the selected features.
        """
        selected_features_idx = np.random.choice(
            len(self.utilities),
            size=self.num_features_to_select,
            replace=False,
            p=self._probability_distribution,
        )

        return selected_features_idx.tolist()
