import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .subproblem_costructor import SubproblemConstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BackboneBase(ABC):
    """
    Base class for backbone algorithms.

    Attributes:
        beta (float): Proportion of screened variables to use in subproblems.
        num_subproblems (int): The number of subproblems to create.
        threshold (float): Threshold for determining feature significance.
        num_iterations (int): Number of iterations.
        screen_selector (ScreenSelectorBase): An instance of a screen selector class.
        exact_solver (ExactSolverBase): An instance of an exact solver class.
        heuristic_solver (HeuristicSolverBase): An instance of a heuristic solver class.
        variables_exact_idx (List[int]): Indices of variables selected for the exact solver.
    """

    def __init__(
        self,
        beta: float = 0.5,
        num_subproblems: int = 5,
        threshold: float = 0.001,
        num_iterations: int = 10,
        **kwargs,
    ):
        """
        Initializes the BackboneBase with specific parameters.

        Args:
            beta (float): Proportion of screened variables to use in subproblems.
            num_subproblems (int): The number of subproblems to create.
            threshold (float, optional): Threshold for determining feature significance.
            num_iterations (int, optional): Number of iterations.
        """
        self.beta = beta
        self.num_subproblems = num_subproblems
        self.threshold = threshold
        self.num_iterations = num_iterations
        self.screen_selector = None
        self.exact_solver = None
        self.heuristic_solver = None
        self.variables_exact_idx = None
        self.init_parameters = kwargs
        self.set_solvers(**kwargs)

    @abstractmethod
    def set_solvers(self, **kwargs):
        """
        Initialize and set the solvers of the Backbone algorithm.
        """

    @abstractmethod
    def preprocessing_backbone(self, X_selected: np.ndarray) -> np.ndarray:
        """Preprocess data specific to the learning method during backbone construction."""

    @abstractmethod
    def set_utilities(self, X: np.ndarray) -> np.ndarray:
        """Set utilities based on the learning method."""

    @abstractmethod
    def utilize_variables(
        self, X_selected: np.ndarray, variables_exact_idx: List[int]
    ) -> np.ndarray:
        """Utilize variables based on the learning method."""

    @abstractmethod
    def preprocessing_predict(self, X: np.ndarray, variables_exact_idx: List[int]) -> np.ndarray:
        """Preprocess data for making predictions based on the learning method."""

    @abstractmethod
    def get_relevant_variables(
        self, feature_idx: List[int], threshold: float = None
    ) -> List[Tuple[int, int]]:
        """
        Abstract method to retrieve relevant variables based on the learning type.

        This method should be implemented in subclasses to handle the specifics of
        identifying relevant variables in both supervised and unsupervised learning contexts.

        Args:
            feature_idx (List[int]): List of feature indices to consider.
            threshold (float, optional): A threshold value used for variable selection,
                                         applicable in some supervised learning scenarios.

        Returns:
            List[Tuple[int, int]]: A list of tuples where each tuple represents a pair of indices
                                   of relevant variables. The structure of these tuples and what they
                                   represent may vary between supervised and unsupervised learning.
        """

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Run the backbone method using the specified screen selector, exact solver, and heuristic solver.

        Args:
            X (np.ndarray): Feature dataset.
            y (np.ndarray): Target values.
        """
        X_selected, utilities = self._perform_screen_selection(X, y)
        self.variables_exact_idx = self._construct_backbone(X_selected, utilities, y)
        self._fit_exact_solver(X_selected, y)

    def _perform_screen_selection(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform screen selection if a screen selector is provided.

        Args:
            X (np.ndarray): Feature dataset.
            y (np.ndarray): Target values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The selected features and their utilities.
        """
        if self.screen_selector:
            logging.info("Screen selection started.")
            X_selected = self.screen_selector.select(X, y)
            utilities = self.screen_selector.utilities[self.screen_selector.indices_keep]
            logging.info(
                f"Number of variables included in the heuristic solver: {X_selected.shape[1]}"
            )
        else:
            logging.info("Screen selection skipped. Using all features.")
            X_selected = X
            utilities = self.set_utilities(X)

        return X_selected, utilities

    def _construct_backbone(
        self, X_selected: np.ndarray, utilities: np.ndarray, y: np.ndarray = None
    ) -> List:
        """
        Construct the backbone using a heuristic solver, or return all features if the heuristic solver is not provided.

        Args:
            X_selected (np.ndarray): Selected feature dataset after screen selection.
            utilities (np.ndarray): Utilities of the selected features.
            y (np.ndarray): Target values.

        Returns:
            List: The indices of the variables selected by the backbone or all feature indices if no heuristic solver is provided.
        """
        if not self.heuristic_solver:
            logging.info("Heuristic solver not provided. Using all features for the exact solver.")
            return list(range(X_selected.shape[1]))

        X_selected = self.preprocessing_backbone(X_selected)
        logging.info(
            f"Backbone construction with heuristic solver started for iterations:{self.num_iterations}, subproblems:{self.num_subproblems} , and beta:{self.beta}"
        )
        backbone_sets = []
        for iter in range(self.num_iterations):
            logging.info(f"Iteration {iter + 1} started.")
            constructor = SubproblemConstructor(utilities, self.beta, self.num_subproblems)
            subproblems = constructor.construct_subproblems()
            for feature_idx in subproblems:
                feature_idx.sort()
                subset = X_selected[:, feature_idx]
                self.heuristic_solver.__init__(**self.init_parameters)
                subset = self.preprocessing_backbone(subset)
                self.heuristic_solver.fit(subset, y)
                rel_variables_global = self.get_relevant_variables(feature_idx, self.threshold)
                backbone_sets.append(rel_variables_global)
            logging.info(f"Iteration {iter + 1} completed.")
        backbone_set = self.build_backbone_set(backbone_sets)
        if self.screen_selector is None:
            logging.info(f"Backbone set idx: {backbone_set}")
        else:
            logging.info(f"Backbone set idx: {self.screen_selector.indices_keep[backbone_set]}")
        return backbone_set

    def _fit_exact_solver(self, X_selected: np.ndarray, y: np.ndarray):
        """
        Fit the exact solver with the selected variables from the backbone.

        Args:
            X_selected (np.ndarray): Selected feature dataset after screen selection.
            y (np.ndarray): Target values.
        """
        # logging.info(f"Backbone set constructed with variables: {self.variables_exact_idx}")
        X_selected_exact = self.utilize_variables(X_selected, self.variables_exact_idx)
        logging.info(
            f"Number of variables included in the exact solver: {X_selected_exact.shape[1]}"
        )
        self.exact_solver.fit(X_selected_exact, y)
        logging.info("Exact problem solved.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted exact solver model.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        X_pred = self.preprocessing_predict(X)
        return self.exact_solver.predict(X_pred)
