import numpy as np
import pytest

from backbone_learn.backbone.backbone_decision_tree import BackboneDecisionTree
from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree
from backbone_learn.heuristic_solvers.cart_decision_tree import CARTDecisionTree


@pytest.fixture
def sample_data():
    # Create a simple dataset for testing
    X = np.random.rand(50, 2)  # 50 samples, 2 features
    y = np.random.randint(0, 2, 50)  # Binary target
    return X, y


def test_initialization():
    """Test initialization of BackboneDecisionTree."""
    backbone = BackboneDecisionTree()
    if not isinstance(backbone, BackboneDecisionTree):
        raise AssertionError("Initialization of BackboneDecisionTree failed")


def test_set_solvers(sample_data):
    """Test the set_solvers method with sample data."""
    backbone = BackboneDecisionTree()
    backbone.set_solvers(
        alpha=0.5, depth=3, time_limit=1000, _lambda=0.5, num_threads=1, obj_mode="acc", n_bins=2
    )

    # Test if solvers are set correctly
    if not isinstance(backbone.exact_solver, BendersOCTDecisionTree):
        raise AssertionError("exact_solver is not an instance of BendersOCTDecisionTree")
    if not isinstance(backbone.heuristic_solver, CARTDecisionTree):
        raise AssertionError("heuristic_solver is not an instance of CARTDecisionTree")


def test_feature_screening(sample_data):
    """Test the feature screening process."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(alpha=0.5)
    screened_features = backbone.screen_selector.select(X, y)

    # Test that the number of features after screening is correct
    if not (0 < screened_features.shape[1] <= X.shape[1]):
        raise AssertionError("Feature screening did not return correct number of features")


def test_exact_solver_integration(sample_data):
    """Test the integration of the exact solver."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(depth=3, time_limit=500, _lambda=0.5)
    backbone.exact_solver.fit(X, y)

    # Asserting model has been fitted
    if backbone.exact_solver.model is None:
        raise AssertionError("exact_solver model has not been fitted")


def test_heuristic_solver_integration(sample_data):
    """Test the integration of the heuristic solver."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers()
    backbone.heuristic_solver.fit(X, y)

    # Asserting model has been fitted and can predict
    predictions = backbone.heuristic_solver.predict(X)
    if len(predictions) != len(y):
        raise AssertionError("Length of predictions does not match length of y")
    if not isinstance(predictions, np.ndarray):
        raise AssertionError("Predictions are not an instance of np.ndarray")
