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
    assert selector.alpha == 1.0
    assert selector.utilities is None
    assert selector.indices_keep is None


def test_utilities_computation(synthetic_data):
    X, y = synthetic_data
    selector = PearsonCorrelationSelector()
    utilities = selector.calculate_utilities(X, y)

    assert utilities is not None
    assert len(utilities) == X.shape[1]


def test_compute_mean():
    array = np.array([1, 2, 3, 4, 5])
    mean = PearsonCorrelationSelector.compute_mean(array)
    assert mean == np.mean(array)


def test_compute_std():
    array = np.array([1, 2, 3, 4, 5])
    std = PearsonCorrelationSelector.compute_std(array)
    assert std == np.std(array)


def test_compute_covariance():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = PearsonCorrelationSelector.compute_covariance(x, y, x_mean, y_mean)
    expected_covariance = np.mean((x - x_mean) * (y - y_mean))
    assert covariance == expected_covariance


def test_select_with_custom_alpha(synthetic_data):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    alpha = 0.5
    selector = PearsonCorrelationSelector(alpha=alpha)
    X_selected = selector.select(X, y)

    expected_features = int(alpha * X.shape[1])
    assert X_selected.shape[1] == expected_features


def test_select_indices():
    utilities = np.array([0.1, 0.5, 0.2])
    num_keep = 2
    selected_indices = PearsonCorrelationSelector.select_indices(utilities, num_keep)

    assert len(selected_indices) == num_keep
    assert np.array_equal(selected_indices, np.array([1, 2]))  # Indices of top two utilities
