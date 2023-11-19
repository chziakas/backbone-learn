import numpy as np
import pytest

from backbone_learn.heuristic_solvers.kmeans_solver import KMeansSolver


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


def test_initialization():
    """Test initialization of KMeansSolver."""
    solver = KMeansSolver(n_clusters=3)
    assert solver.n_clusters == 3
    assert solver._model is None


def test_fit(sample_data):
    """Test fitting the KMeans model."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    assert solver._model is not None
    assert len(solver.cluster_centers) == 2


def test_compute_cluster_centers(sample_data):
    """Test computation of cluster centers."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    assert solver.cluster_centers.shape == (2, 2)


def test_compute_wcss(sample_data):
    """Test computation of WCSS."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    wcss = solver._compute_wcss(sample_data)
    assert isinstance(wcss, float)
    assert wcss > 0


def test_get_relevant_variables(sample_data):
    """Test identification of relevant variables."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    relevant_vars = solver.get_relevant_variables()
    assert isinstance(relevant_vars, list)
    assert all(isinstance(pair, tuple) for pair in relevant_vars)


def test_compute_silhouette_score(sample_data):
    """Test computation of silhouette score."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    score = solver._compute_silhouette_score(sample_data)
    assert isinstance(score, float)
