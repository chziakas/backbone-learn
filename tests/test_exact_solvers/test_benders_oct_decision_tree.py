import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree


def test_default_initialization():
    tree = BendersOCTDecisionTree()
    assert isinstance(tree.est_X, KBinsDiscretizer)
    assert isinstance(tree.enc, OneHotEncoder)
    assert not tree.is_data_fit


def test_preprocess_features_with_numpy():
    tree = BendersOCTDecisionTree()
    X = np.random.rand(10, 2)  # Sample data
    tree.fit_preprocessors(X)  # Fit preprocessors first
    X_transformed = tree.preprocess_features(X)
    assert X_transformed.shape == (10, 2)  # Expected shape


def test_fit_preprocessors():
    tree = BendersOCTDecisionTree()
    X_train = np.random.rand(10, 2)  # Sample training data
    tree.fit_preprocessors(X_train)
    assert tree.est_X.n_bins_ is not None  # est_X should be fitted
    assert tree.enc.categories_ is not None  # enc should be fitted
