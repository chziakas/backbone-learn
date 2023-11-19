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

    assert reg.model is not None


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
    assert len(significant_features) > 0
    assert all(abs(reg.model.coef_[idx]) > threshold for idx in significant_features)


def test_lasso_regression_predict():
    X_train, y_train = make_regression(n_samples=100, n_features=4, random_state=42)
    X_test, y_test = make_regression(n_samples=20, n_features=4, random_state=42)

    model = LassoRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
