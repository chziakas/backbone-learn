import numpy as np
from sklearn.datasets import make_blobs

from backbone_learn.exact_solvers.mio_clustering import (  # Import your MIOClustering class here
    MIOClustering,
)


def test_calculate_distances():
    X = np.array([[0, 0], [3, 4]])
    clustering = MIOClustering(n_clusters=2)
    distances = clustering._calculate_distances(X)
    assert distances.shape == (2, 2, 2)
    assert 4.5 <= distances[0, 1, 0] <= 5.5


def test_fit_predict():
    X, _ = make_blobs(n_samples=10, n_features=2, centers=3, random_state=42)
    clustering = MIOClustering(n_clusters=3)
    clustering.fit(X)
    assignments = clustering.predict(X)
    assert len(assignments) == 10
    assert len(set(assignments)) == 3  # Assuming 2 clusters are identified
