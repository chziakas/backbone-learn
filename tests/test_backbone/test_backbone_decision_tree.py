import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from backbone_learn.backbone.backbone_decision_tree import BackboneDecisionTree
from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree
from backbone_learn.heuristic_solvers.cart_decision_tree import CARTDecisionTree


@pytest.fixture
def sample_data():
    # Create a simple dataset for testing
    X = np.random.rand(50, 2)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 50)  # Binary target
    return X, y


def test_initialization():
    """Test initialization of BackboneDecisionTree."""
    backbone = BackboneDecisionTree()
    assert isinstance(backbone, BackboneDecisionTree)


def test_set_solvers(sample_data):
    """Test the set_solvers method with sample data."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(
        alpha=0.5, depth=3, time_limit=1000, _lambda=0.5, num_threads=1, obj_mode="acc", n_bins=2
    )

    # Test if solvers are set correctly
    assert isinstance(backbone.exact_solver, BendersOCTDecisionTree)
    assert isinstance(backbone.heuristic_solver, CARTDecisionTree)


def test_feature_screening(sample_data):
    """Test the feature screening process."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(alpha=0.5)
    screened_features = backbone.screen_selector.select(X, y)

    # Assert that the number of features after screening is correct
    assert 0 < screened_features.shape[1] <= X.shape[1]


def test_exact_solver_integration(sample_data):
    """Test the integration of the exact solver."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(depth=3, time_limit=500, _lambda=0.5)
    backbone.exact_solver.fit(X, y)

    # Asserting model has been fitted
    assert backbone.exact_solver.model is not None


def test_heuristic_solver_integration(sample_data):
    """Test the integration of the heuristic solver."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers()
    backbone.heuristic_solver.fit(X, y)

    # Asserting model has been fitted and can predict
    predictions = backbone.heuristic_solver.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)


def test_end_to_end_pipeline(sample_data):
    """Test the entire pipeline from feature screening to model fitting."""
    X, y = sample_data
    backbone = BackboneDecisionTree()
    backbone.set_solvers(
        alpha=0.5, depth=3, time_limit=1000, _lambda=0.5, num_threads=1, obj_mode="acc", n_bins=2
    )

    screened_X = backbone.screen_selector.select(X, y)
    backbone.exact_solver.fit(screened_X, y)
    predictions = backbone.exact_solver.predict(screened_X)

    # Check if predictions are reasonable
    assert len(predictions) == len(y)
    assert accuracy_score(y, predictions) >= 0.5  # Or any other relevant threshold
