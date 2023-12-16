# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

import numpy as np
from sklearn.datasets import make_regression

from backbone_learn.heuristic_solvers.lasso_regression import LassoRegression


def test_fit():
    # Creating a small synthetic dataset
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    reg = LassoRegression()
    reg.fit(X, y)

    if reg.model is None:
        raise AssertionError("LassoRegression model not initialized after fit")


def test_get_significant_features():
    # Create a synthetic dataset
    np.random.seed(0)  # Ensuring reproducibility
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    # Create a response variable with some random noise
    y = np.dot(X, np.array([1, 0.5, -1, 2, 0, 0, 0, 0, 0, 0])) + np.random.randn(100)

    # Initialize and fit the SparseRegression model
    reg = LassoRegression()
    reg.fit(X, y)

    # Set a threshold for significant features
    threshold = 0.1
    significant_features = reg.get_relevant_variables(threshold)

    # Check if the method identifies the correct features
    if len(significant_features) == 0:
        raise AssertionError("No significant features identified")
    if not all(abs(reg.model.coef_[idx]) > threshold for idx in significant_features):
        raise AssertionError("Identified significant features do not meet threshold")


def test_lasso_regression_predict():
    X_train, y_train = make_regression(n_samples=100, n_features=4, random_state=42)
    X_test, _ = make_regression(n_samples=20, n_features=4, random_state=42)

    model = LassoRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if len(predictions) != len(X_test):
        raise AssertionError("Number of predictions does not match number of test samples")
