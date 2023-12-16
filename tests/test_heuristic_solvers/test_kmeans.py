# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

import numpy as np
import pytest

from backbone_learn.heuristic_solvers.kmeans_solver import KMeansSolver


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


def test_fit(sample_data):
    """Test fitting the KMeans model."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    if solver._model is None:
        raise AssertionError("KMeans model not initialized after fit")
    if len(solver.cluster_centers) != 2:
        raise AssertionError("Number of cluster centers is not 2 as expected")


def test_compute_cluster_centers(sample_data):
    """Test computation of cluster centers."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    if solver.cluster_centers.shape != (2, 2):
        raise AssertionError("Cluster centers shape is not (2, 2) as expected")


def test_compute_wcss(sample_data):
    """Test computation of WCSS."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    wcss = solver._compute_wcss(sample_data)
    if not isinstance(wcss, float):
        raise AssertionError("WCSS is not a float value")
    if wcss <= 0:
        raise AssertionError("WCSS is not greater than 0")


def test_get_relevant_variables(sample_data):
    """Test identification of relevant variables."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    relevant_vars = solver.get_relevant_variables()
    if not isinstance(relevant_vars, list):
        raise AssertionError("Relevant variables not returned as a list")
    if not all(isinstance(pair, tuple) for pair in relevant_vars):
        raise AssertionError("Elements in relevant variables are not tuples")


def test_compute_silhouette_score(sample_data):
    """Test computation of silhouette score."""
    solver = KMeansSolver(n_clusters=2)
    solver.fit(sample_data)
    score = solver._compute_silhouette_score(sample_data)
    if not isinstance(score, float):
        raise AssertionError("Silhouette score is not a float value")
