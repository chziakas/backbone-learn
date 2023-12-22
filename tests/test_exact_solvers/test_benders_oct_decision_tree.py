# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree


def test_default_initialization():
    tree = BendersOCTDecisionTree()
    if not isinstance(tree.est_X, KBinsDiscretizer):
        raise AssertionError("tree.est_X is not an instance of KBinsDiscretizer")
    if not isinstance(tree.enc, OneHotEncoder):
        raise AssertionError("tree.enc is not an instance of OneHotEncoder")
    if tree.is_data_fit:
        raise AssertionError("tree.is_data_fit should be False on default initialization")


def test_preprocess_features_with_numpy():
    tree = BendersOCTDecisionTree()
    X = np.random.rand(10, 2)  # Sample data
    tree.fit_preprocessors(X)  # Fit preprocessors first
    X_transformed = tree.preprocess_features(X)
    if X_transformed.shape != (10, 2):  # Expected shape
        raise AssertionError("X_transformed does not have the expected shape")


def test_fit_preprocessors():
    tree = BendersOCTDecisionTree()
    X_train = np.random.rand(10, 2)  # Sample training data
    tree.fit_preprocessors(X_train)
    if tree.est_X.n_bins_ is None:  # est_X should be fitted
        raise AssertionError("tree.est_X.n_bins_ is not set after fitting preprocessors")
    if tree.enc.categories_ is None:  # enc should be fitted
        raise AssertionError("tree.enc.categories_ is not set after fitting preprocessors")


def test_predict_after_fitting_preprocessors():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = BendersOCTDecisionTree()
    model.fit(X, y)
    predictions = model.predict(X)
    if len(predictions) != len(y):
        raise AssertionError("Prediction length mismatch with input")


def test_predict_after_fitting_preprocessors_fit():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_binary = (X > 0.5).astype(int)  # Convert to binary (0 or 1)
    model = BendersOCTDecisionTree(is_data_fit=True)  # Set is_data_fit to True for testing
    model.fit(X_binary, y)
    predictions = model.predict(X_binary)
    if len(predictions) != len(y):
        raise AssertionError("Prediction length mismatch with input")


def test_predict_after_fitting_preprocessors():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = BendersOCTDecisionTree(is_data_fit=False)
    model.fit(X, y)
    predictions = model.predict(X)
    if len(predictions) != len(y):
        raise AssertionError("Prediction length mismatch with input")
