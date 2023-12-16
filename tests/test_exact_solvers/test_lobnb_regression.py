# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

from sklearn.datasets import make_regression

from backbone_learn.exact_solvers.lobnb_regression import L0BnBRegression


def test_l0bnb_regression():
    # Generate a synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42
    )

    # Initialize and fit the L0BnBRegression model
    l0bnb_reg = L0BnBRegression(lambda_2=0.01, max_nonzeros=10)
    l0bnb_reg.fit(X, y)

    # Test that solutions are found
    if len(l0bnb_reg.model.coefficients) == 0:
        raise AssertionError("L0BnBRegression model did not find any coefficients")
    if l0bnb_reg.model is None:
        raise AssertionError("L0BnBRegression model is not initialized")

    # Test predictions
    predictions = l0bnb_reg.predict(X)
    if len(predictions) != len(y):
        raise AssertionError("Number of predictions does not match number of samples")
