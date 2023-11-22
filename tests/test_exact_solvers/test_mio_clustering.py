import numpy as np
from sklearn.datasets import make_blobs

from backbone_learn.exact_solvers.mio_clustering import MIOClustering


def test_calculate_distances():
    X = np.array([[0, 0], [3, 4]])
    clustering = MIOClustering(n_clusters=2)
    distances = clustering._calculate_distances(X)
    if distances.shape != (2, 2, 2):
        raise AssertionError("Distances array does not have the expected shape (2, 2, 2)")
    if not (4.5 <= distances[0, 1, 0] <= 5.5):
        raise AssertionError("Distance calculation is not within the expected range (4.5 to 5.5)")


def test_fit_predict():
    X, _ = make_blobs(n_samples=10, n_features=2, centers=3, random_state=42)
    clustering = MIOClustering(n_clusters=3)
    clustering.fit(X)
    assignments = clustering.predict(X)
    if len(assignments) != 10:
        raise AssertionError("Number of assignments does not match number of samples")
    if len(set(assignments)) != 3:  # Assuming 3 clusters are identified
        raise AssertionError(
            "Number of unique assignments does not match expected number of clusters"
        )
