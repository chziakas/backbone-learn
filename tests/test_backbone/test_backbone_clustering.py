# Copyright (c) 2023 Vassilis Digalakis Jr, Christos Ziakas
# Licensed under the MIT License.

import pytest
from sklearn.datasets import make_blobs

from backbone_learn.backbone.backbone_clustering import BackboneClustering
from backbone_learn.heuristic_solvers.kmeans_solver import KMeansSolver


@pytest.fixture
def sample_data():
    X, _ = make_blobs(n_samples=10, centers=2, n_features=3, random_state=17)
    return X


def test_backbone_clustering(sample_data):
    n_clusters = 5
    backbone_model = BackboneClustering(
        beta=1.0, num_subproblems=1, num_iterations=1, n_clusters=n_clusters, time_limit=3600
    )
    backbone_model.fit(sample_data)

    # Test if the model has fitted
    if backbone_model.exact_solver is None:
        raise AssertionError("Backbone model's exact solver is not initialized")
    if backbone_model.heuristic_solver is None:
        raise AssertionError("Backbone model's heuristic solver is not initialized")

    # Test constraints are applied
    for (i, j) in backbone_model.exact_solver.ls_pairs_diff_cluster:
        for k in range(n_clusters):
            y_sum = backbone_model.exact_solver.y[i, k] + backbone_model.exact_solver.y[j, k]
            if y_sum >= 2:
                raise AssertionError("Constraint on y_sum violated")
            if backbone_model.exact_solver.z[i, j, k] != 0.0:
                raise AssertionError("Constraint on z violated")

    # Test silhouette scores
    if not (0 <= backbone_model.exact_solver.silhouette_score <= 1):
        raise AssertionError("Exact solver's silhouette score out of range")
    if not (0 <= backbone_model.heuristic_solver.silhouette_score <= 1):
        raise AssertionError("Heuristic solver's silhouette score out of range")


def test_kmeans_solver(sample_data):
    n_clusters = 5
    heuristic_model = KMeansSolver(n_clusters=n_clusters)
    heuristic_model.fit(sample_data)

    # Test if the model has fitted
    if heuristic_model.model is None:
        raise AssertionError("KMeans solver model is not initialized")

    # Test silhouette scores
    if not (0 <= heuristic_model.silhouette_score <= 1):
        raise AssertionError("KMeans solver's silhouette score out of range")

    # Test WCSS
    if heuristic_model.wcss < 0:
        raise AssertionError("KMeans solver's WCSS is negative")
