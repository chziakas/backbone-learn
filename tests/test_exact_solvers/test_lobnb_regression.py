from sklearn.datasets import make_regression

from backbone_learn.exact_solvers.lobnb_regression import L0BnBRegression

"""
def test_l0bnb_regression_doc():
    # Generate synthetic dataset
    X, y, b_true = gen_synthetic(n=1000, p=10000, supp_size=10)

    # Check that the true coefficient vector is sparse
    nonzero_indices_true = np.nonzero(b_true)[0]
    assert len(nonzero_indices_true) == 10

    # Run L0BnB
    sols = fit_path(X, y, lambda_2=0.01, max_nonzeros=10)

    # Assert that sols is not empty
    assert len(sols) > 0

    # Inspect a specific solution
    b_estimated = sols[4]["B"]  # a numpy array
    intercept = sols[4]["B0"]

    # Check the nonzero indices in the estimated coefficients
    nonzero_indices_estimated = np.nonzero(b_estimated)[0]
    assert len(nonzero_indices_estimated) <= 10

    # Check if the nonzero indices in b_estimated are consistent with b_true
    assert set(nonzero_indices_true).issubset(set(nonzero_indices_estimated))

    # Check predictions
    y_estimated = np.dot(X, b_estimated) + intercept
    assert y_estimated.shape == y.shape
"""


def test_l0bnb_regression():
    # Generate a synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42
    )

    # Initialize and fit the L0BnBRegression model
    l0bnb_reg = L0BnBRegression(lambda_2=0.01, max_nonzeros=10)
    l0bnb_reg.fit(X, y)

    # Assert that solutions are found
    assert len(l0bnb_reg.model.coefficients) > 0
    assert l0bnb_reg.model is not None

    # Test predictions
    predictions = l0bnb_reg.predict(X)
    assert len(predictions) == len(y)
