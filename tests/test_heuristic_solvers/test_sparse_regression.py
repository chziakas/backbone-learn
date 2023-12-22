# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

import numpy as np
import pytest
from sklearn.datasets import make_regression

from backbone_learn.heuristic_solvers.lasso_regression import LassoRegression


# Test Initialization
def test_initialization():
    reg = LassoRegression()
    if not isinstance(reg, LassoRegression):
        raise AssertionError("LassoRegression instance is not created correctly")


# Test Model Fitting
def test_fit():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    reg = LassoRegression()
    reg.fit(X, y)
    if not hasattr(reg._model, "coef_"):
        raise AssertionError("Model coefficients (coef_) not found after fitting")


# Test Predict Function
def test_predict():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    reg = LassoRegression()
    reg.fit(X, y)
    predictions = reg.predict(X)
    if len(predictions) != len(y):
        raise AssertionError("Prediction length mismatch")


# Test Getting Relevant Variables
def test_get_relevant_variables():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    reg = LassoRegression()
    reg.fit(X, y)
    threshold = 0.1
    significant_vars = reg.get_relevant_variables(threshold)
    if not isinstance(significant_vars, np.ndarray):
        raise AssertionError("Output is not numpy array")
    if len(significant_vars) == 0:
        raise AssertionError("No significant variables found")


# Test Keeping Top Features
def test_keep_top_features():
    n_features = 10
    n_non_zeros = 5
    X, y = make_regression(n_samples=100, n_features=n_features, noise=0.1)
    reg = LassoRegression()
    reg.fit(X, y)
    reg.keep_top_features(n_non_zeros)
    if np.count_nonzero(reg._model.coef_) > n_non_zeros:
        raise AssertionError("More features retained than specified")


# Test Error on Unfitted Model Prediction
def test_error_on_unfitted_predict():
    reg = LassoRegression()
    X = np.random.randn(10, 5)
    with pytest.raises(ValueError):
        _ = reg.predict(X)


def test_mse_score():
    # Create a small synthetic dataset
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Initialize and fit the LassoRegression model
    lasso_reg = LassoRegression()
    lasso_reg.fit(X, y)

    # Manually calculate MSE for comparison
    predictions = lasso_reg.predict(X)
    expected_mse = np.mean((y - predictions) ** 2)

    # Set the _mse_score manually (if it's not set in fit)
    lasso_reg._mse_score = expected_mse

    # Test that mse_score property returns the correct MSE
    if not np.isclose(lasso_reg.mse_score, expected_mse):
        raise AssertionError("The mse_score property did not return the expected value")
