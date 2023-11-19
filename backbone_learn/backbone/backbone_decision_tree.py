from ..exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree
from ..heuristic_solvers.cart_decision_tree import CARTDecisionTree
from ..screen_selectors.pearson_correlation_selector import PearsonCorrelationSelector
from .backbone_supervised import BackboneSupervised


class BackboneDecisionTree(BackboneSupervised):
    """
    Specific implementation of the Backbone method for sparse regression.

    This class combines Pearson correlation for feature screening, L0BnB for exact solving, and Lasso for heuristic solving to construct a sparse regression model.

    Inherits from:
        BackboneBase (ABC): The abstract base class for backbone algorithms.
    """

    def set_solvers(
        self,
        alpha=0.5,
        depth=3,
        time_limit=1000,
        _lambda=0.5,
        num_threads=None,
        obj_mode="acc",
        n_bins=2,
    ):
        """
        Initializes the sparse regression method with specified components.

        Args:
            alpha (float): Proportion of features to retain after screening. Defaults to 0.5.
            depth (int, optional): Depth of BendersOCT tree. Defaults to 3.
            time_limit (int): Time limit for the optimization process.
            _lambda (float): Regularization parameter.
            num_threads (int or None): Number of threads for parallel processing.
            obj_mode (str): Objective mode, e.g., 'acc' for accuracy.
            n_bins (int): Number of bins for KBinsDiscretizer. Defaults to 2.
        """
        self.screen_selector = PearsonCorrelationSelector(alpha)
        self.exact_solver = BendersOCTDecisionTree(
            depth=depth,
            time_limit=time_limit,
            _lambda=_lambda,
            num_threads=num_threads,
            obj_mode=obj_mode,
            n_bins=n_bins,
        )
        self.heuristic_solver = CARTDecisionTree()
