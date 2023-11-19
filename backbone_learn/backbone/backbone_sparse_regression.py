from ..exact_solvers.lobnb_regression import L0BnBRegression
from ..heuristic_solvers.lasso_regression import LassoRegression
from ..screen_selectors.pearson_correlation_selector import PearsonCorrelationSelector
from .backbone_supervised import BackboneSupervised


class BackboneSparseRegression(BackboneSupervised):
    """
    Specific implementation of the Backbone method for sparse regression.

    This class combines Pearson correlation for feature screening, L0BnB for exact solving, and Lasso for heuristic solving to construct a sparse regression model.

    Inherits from:
        BackboneBase (ABC): The abstract base class for backbone algorithms.
    """

    def set_solvers(
        self, alpha: float = 0.5, lambda_2: float = 0.001, max_nonzeros: int = 10, time_limit=1000
    ):
        """
        Initializes the sparse regression method with specified components.

        Args:
            alpha (float): Proportion of features to retain after screening.
            lambda_2 (float, optional): Regularization parameter lambda_2 in the L0BnB model, controlling the trade-off between the model's complexity and fit. Defaults to 0.001.
            max_nonzeros (int, optional): Maximum number of non-zero coefficients the model is allowed to have, enforcing sparsity. Defaults to 10.
            time_limit (int): Time limit for the optimization process.
        """
        self.screen_selector = PearsonCorrelationSelector(alpha)
        self.exact_solver = L0BnBRegression(lambda_2, max_nonzeros, time_limit)
        self.heuristic_solver = LassoRegression()
