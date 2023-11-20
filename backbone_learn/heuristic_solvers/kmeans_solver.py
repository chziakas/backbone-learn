import random
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .heauristic_solver_base import HeuristicSolverBase


class KMeansSolver(HeuristicSolverBase):
    """
    A heuristic solver that applies KMeans clustering to identify relevant instances.
    """

    def __init__(self, n_clusters: int = 10, **kwargs) -> None:
        """
        Initializes the KMeansHeuristicSolver with a specified number of clusters.
        Args:
            n_clusters (int): The number of clusters to form.
        """
        self.n_clusters: int = n_clusters
        self._model: Optional[KMeans] = None
        self.wcss = None

    def _compute_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Extract cluster means after fitting the model.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: An array of cluster means.
        """
        # Initialize an array to store means
        cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(X.shape[0]):
            for k in range(self.n_clusters):
                cluster_points = []  # List to store data points assigned to the current cluster
                if self._model.labels_[i] == k:
                    cluster_points.append(X[i, :])
                if cluster_points:
                    cluster_centers[k] = np.mean(cluster_points, axis=0)

        return cluster_centers

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 0.0001,
        random_state: int = -1,
    ) -> None:
        """
        Applies KMeans clustering to the data with customizable hyperparameters.
        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target vector (not used in clustering).
            init (str): Method for initialization.
            n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers.
            random_state (int): Determines random number generation for centroid initialization.
        """
        if X.shape[0] < self.n_clusters:
            self.n_clusters = X.shape[0]

        if random_state == -1:
            random_state = random.randint(1, 100)
        # If n_clusters is not specified, use the class attribute
        self._model = KMeans(
            n_clusters=self.n_clusters,
            init="random",
            n_init="auto",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self._model.fit(X)
        self.cluster_centers = self._compute_cluster_centers(X)
        self.wcss = self._compute_wcss(X)
        self.silhouette_score = self._compute_silhouette_score(X)

    def get_relevant_variables(self) -> List[Tuple[int, int]]:
        """
        Identifies tuples of instance indices that are not in the same cluster.
        Returns:
            List of tuples: Each tuple contains indices of instances not in the same cluster.
        """

        n = len(self._model.labels_)
        grid_x, grid_y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = self._model.labels_[grid_x] != self._model.labels_[grid_y]
        upper_triangle_mask = np.triu(mask, k=1)
        i_indices, j_indices = np.where(upper_triangle_mask)
        different_pairs = [(min(i, j), max(i, j)) for i, j in zip(i_indices, j_indices)]
        return different_pairs

    def _compute_wcss(self, X: np.ndarray) -> float:
        """
        Mthod to calculate the Within-Cluster Sum of Squares (WCSS).

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            float: The WCSS value.
        """
        if self._model is None or not hasattr(self._model, "cluster_centers_"):
            raise ValueError("The KMeans model must be fitted before calculating WCSS.")
        wcss = 0.0
        cluster_labels_pred = self._model.labels_

        for cluster_idx in range(self.n_clusters):
            cluster_points = X[cluster_labels_pred == cluster_idx]
            wcss += np.sum((cluster_points - self.cluster_centers[cluster_idx]) ** 2)

        return wcss

    def _compute_silhouette_score(self, X: np.ndarray) -> float:
        """ """
        from sklearn.metrics import silhouette_score

        silhouette_avg = silhouette_score(X, self._model.labels_)
        return silhouette_avg
