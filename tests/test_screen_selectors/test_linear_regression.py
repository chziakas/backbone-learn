import numpy as np

from backbone_learn.screen_selectors.linear_regression_selector import LinearRegressionSelector


def test_linear_regression_selector():
    # Test data: a simple linear relationship
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    # Initialize LinearRegressionSelector
    selector = LinearRegressionSelector()
    utilities = selector.calculate_utilities(X, y)

    # Expected coefficients: [1, 2] (without considering the intercept)
    expected_utilities = np.array([2])

    # Verify that calculated utilities match expected values
    if not np.allclose(utilities, expected_utilities):
        raise AssertionError(f"Expected utilities {expected_utilities}, got {utilities}")


# Test Utilities Calculation with Singular Matrix
def test_utilities_with_singular_matrix():
    selector = LinearRegressionSelector()
    X = np.ones((10, 3))  # Singular matrix (not invertible)
    y = np.random.rand(10)
    utilities = selector.calculate_utilities(X, y)

    if not np.all(utilities == 0):
        raise AssertionError("Utilities should be zero for singular matrix")
