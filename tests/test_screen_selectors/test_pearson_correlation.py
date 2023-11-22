import numpy as np
import pytest

from backbone_learn.screen_selectors.pearson_correlation_selector import PearsonCorrelationSelector


@pytest.fixture
def synthetic_data():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return X, y


def test_initialization():
    selector = PearsonCorrelationSelector()
    if selector.alpha != 1.0:
        raise AssertionError("Selector alpha not initialized to 1.0")
    if selector.utilities is not None:
        raise AssertionError("Selector utilities not initialized as None")
    if selector.indices_keep is not None:
        raise AssertionError("Selector indices_keep not initialized as None")


def test_utilities_computation(synthetic_data):
    X, y = synthetic_data
    selector = PearsonCorrelationSelector()
    utilities = selector.calculate_utilities(X, y)

    if utilities is None:
        raise AssertionError("Utilities computation returned None")
    if len(utilities) != X.shape[1]:
        raise AssertionError("Incorrect number of utilities computed")


def test_compute_mean():
    array = np.array([1, 2, 3, 4, 5])
    mean = PearsonCorrelationSelector.compute_mean(array)
    if mean != np.mean(array):
        raise AssertionError("Computed mean does not match expected mean")


def test_compute_std():
    array = np.array([1, 2, 3, 4, 5])
    std = PearsonCorrelationSelector.compute_std(array)
    if std != np.std(array):
        raise AssertionError("Computed standard deviation does not match expected value")


def test_compute_covariance():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = PearsonCorrelationSelector.compute_covariance(x, y, x_mean, y_mean)
    expected_covariance = np.mean((x - x_mean) * (y - y_mean))
    if covariance != expected_covariance:
        raise AssertionError("Computed covariance does not match expected value")


def test_select_with_custom_alpha(synthetic_data):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    alpha = 0.5
    selector = PearsonCorrelationSelector(alpha=alpha)
    X_selected = selector.select(X, y)

    expected_features = int(alpha * X.shape[1])
    if X_selected.shape[1] != expected_features:
        raise AssertionError("Selected features do not match expected number based on alpha")


def test_select_indices():
    utilities = np.array([0.1, 0.5, 0.2])
    num_keep = 2
    selected_indices = PearsonCorrelationSelector.select_indices(utilities, num_keep)

    if len(selected_indices) != num_keep:
        raise AssertionError("Incorrect number of indices selected")
    if not np.array_equal(selected_indices, np.array([1, 2])):
        raise AssertionError("Selected indices do not match expected top two utilities")
