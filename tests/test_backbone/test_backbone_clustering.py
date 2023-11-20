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
    assert backbone_model.exact_solver is not None
    assert backbone_model.heuristic_solver is not None

    # Test constraints are applied
    for (i, j) in backbone_model.exact_solver.ls_pairs_diff_cluster:
        for k in range(n_clusters):
            y_sum = backbone_model.exact_solver.y[i, k] + backbone_model.exact_solver.y[j, k]
            assert y_sum < 2
            assert backbone_model.exact_solver.z[i, j, k] == 0.0

    # Test silhouette scores
    assert 0 <= backbone_model.exact_solver.silhouette_score <= 1
    assert 0 <= backbone_model.heuristic_solver.silhouette_score <= 1


def test_kmeans_solver(sample_data):
    n_clusters = 5
    heuristic_model = KMeansSolver(n_clusters=n_clusters)
    heuristic_model.fit(sample_data)

    # Test if the model has fitted
    assert heuristic_model.model is not None

    # Test silhouette scores
    assert 0 <= heuristic_model.silhouette_score <= 1

    # Test WCSS
    assert heuristic_model.wcss >= 0
