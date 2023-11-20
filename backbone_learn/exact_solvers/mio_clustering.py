from typing import List, Optional, Tuple

import numpy as np
from pulp import PULP_CBC_CMD, LpBinary, LpMinimize, LpProblem, LpVariable, lpSum


class MIOClustering:
    """
    Class for solving clustering problems using Mixed-Integer Optimization.
    """

    def __init__(
        self,
        n_clusters: int = None,
        time_limit: float = 1200,
        constraints: Optional[List[Tuple[int, int]]] = None,
    ):
        self.n_clusters = n_clusters
        self.constraints = constraints
        self.time_limit = time_limit
        self.model = LpProblem("Clustering MIO", LpMinimize)
        self.z = None  # For storing solution for z variables
        self.y = None  # For storing solution for y variables
        self.cluster_means = None

    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    def _initialize_variables(self, num_points: int):
        """
        Initialize the decision variables for the optimization problem.

        Args:
            num_points (int): The number of data points.

        Returns:
            Tuple: A tuple containing the dictionaries of z and y variables.
        """
        z = LpVariable.dicts(
            "z",
            [
                (i, j, k)
                for i in range(num_points - 1)
                for j in range(i + 1, num_points)
                for k in range(self.n_clusters)
            ],
            0,
            1,
            LpBinary,
        )

        y = LpVariable.dicts(
            "y",
            [(i, k) for i in range(num_points) for k in range(self.n_clusters)],
            0,
            1,
            LpBinary,
        )

        return z, y

    def _calculate_distances_noise(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate and return the matrix of pairwise distances with added noise.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:2
            np.ndarray: The matrix of pairwise distances with noise.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        min_dist = np.min(distances[np.nonzero(distances)])
        noise = 0.1 * min_dist * (2 * np.random.rand(X.shape[0], X.shape[0], self.n_clusters) - 1)
        return distances[:, :, np.newaxis] + noise

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate and return the matrix of pairwise distances

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:2
            np.ndarray: The matrix of pairwise distances with noise.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        return np.tile(distances[:, :, np.newaxis], (1, 1, self.n_clusters))

    def _add_constraints(self, num_points: int, z: dict, y: dict, coef: np.ndarray, b: int):
        """
        Add constraints to the optimization model.

        Args:
            num_points (int): The number of data points.
            z (dict): The decision variables representing pair assignments.
            y (dict): The decision variables representing individual assignments.
            coef (np.ndarray): Coefficient matrix for the objective function.
            b (int): Minimum number of points per cluster.
        """
        # Objective

        z_opt, y_opt = self._initialize_variables(num_points)

        if self.constraints:
            for (i, j) in self.constraints:
                for k in range(self.n_clusters):
                    z_opt[i, j, k].setInitialValue(0)
                    z_opt[i, j, k].fixValue()

        self.model += lpSum(
            z_opt[i, j, k] * coef[i, j, k]
            for i in range(num_points - 1)
            for j in range(i + 1, num_points)
            for k in range(self.n_clusters)
        )

        # Each point is assigned to exactly one cluster
        for i in range(num_points):
            self.model += lpSum(y_opt[i, k] for k in range(self.n_clusters)) == 1

        # Each cluster has at least b points
        for k in range(self.n_clusters):
            self.model += lpSum(y_opt[i, k] for i in range(num_points)) >= b

        # Relationship between y and z variables
        for i in range(num_points - 1):
            for j in range(i + 1, num_points):
                for k in range(self.n_clusters):
                    self.model += z_opt[i, j, k] <= y_opt[i, k]
                    self.model += z_opt[i, j, k] <= y_opt[j, k]
                    self.model += z_opt[i, j, k] >= y_opt[i, k] + y_opt[j, k] - 1

        # Exclusion constraints
        if self.constraints:
            for (i, j) in self.constraints:
                for k in range(self.n_clusters):
                    self.model += y_opt[i, k] + y_opt[j, k] <= 1

    def extract_cluster_means(self, X: np.ndarray) -> np.ndarray:
        """
        Extract cluster means after fitting the model.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: An array of cluster means.
        """
        num_points = len(X)
        # Initialize an array to store means
        cluster_means = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = []  # List to store data points assigned to the current cluster
            for i in range(num_points):
                if self.y[i, k] == 1.0:
                    cluster_points.append(X[i])

            if cluster_points:
                cluster_means[k] = np.mean(cluster_points, axis=0)

        return cluster_means

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the model to the given data using Mixed-Integer Optimization.

        Args:
            X (np.ndarray): The input feature matrix.
            y (Optional[np.ndarray]): The target vector (not used in this model).
        """
        num_points = len(X)
        b = int((num_points / self.n_clusters) * 0.1)  # Minimum number of points per cluster

        coef = self._calculate_distances_noise(X)

        self._add_constraints(num_points, self.z, self.y, coef, b)

        solver = PULP_CBC_CMD(timeLimit=self.time_limit, warmStart=True)

        # Solve the problem
        self.model.solve(solver)

        self.y = np.zeros((num_points, self.n_clusters))
        self.z = np.zeros((num_points, num_points, self.n_clusters))

        for v in self.model.variables():
            var_value = v.varValue
            var_name = v.name
            if var_name.startswith("y_"):
                # Parse the indices for y
                i, k = (
                    var_name.replace("(", "")
                    .replace(")", "")
                    .replace("y_", "")
                    .replace(",", "")
                    .split("_")
                )
                i, k = int(i), int(k)
                self.y[i, k] = var_value
            elif var_name.startswith("z_"):
                # Parse the indices for z
                i, j, k = (
                    var_name.replace("(", "")
                    .replace(")", "")
                    .replace("z_", "")
                    .replace(",", "")
                    .split("_")
                )
                i, j, k = int(i), int(j), int(k)
                self.z[i, j, k] = var_value

        # Extract and store cluster means
        self.labels = self._get_cluster_assingments(X.shape[0])
        self.cluster_centers = self._compute_cluster_centers(X)
        self.wcss = self._compute_wcss(X)
        self.silhouette_score = self._compute_silhouette_score(X)

    def _get_cluster_assingments(self, n_rows: int) -> np.ndarray:
        """
        Predict cluster assignments for new data points based on stored cluster means.

        Args:
            new_data (np.ndarray): The new data points for which predictions are to be made.

        Returns:
            np.ndarray: An array of cluster assignments for the new data points.
        """
        cluster_assignments = np.zeros(n_rows, dtype=int)

        for i in range(n_rows):
            cluster_assignments[i] = np.argmax(self.y[i, :])  # np.argmin(distances)
        return cluster_assignments

    def _compute_wcss(self, X: np.ndarray) -> float:
        """
        Compute the Within-Cluster Sum of Squares (WCSS) for the fitted model.

        Args:
            X (np.ndarray): The input feature matrix used for fitting the model.

        Returns:
            float: The computed WCSS value.

        Raises:
            ValueError: If the model has not been fitted yet or if cluster means are not available.
        """

        wcss = 0.0
        cluster_labels_pred = self.labels

        for cluster_idx in range(self.n_clusters):
            cluster_points = X[cluster_labels_pred == cluster_idx]
            wcss += np.sum((cluster_points - self.cluster_centers[cluster_idx]) ** 2)
        return wcss

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
                if self.labels[i] == k:
                    cluster_points.append(X[i, :])
                if cluster_points:
                    cluster_centers[k] = np.mean(cluster_points, axis=0)

        return cluster_centers

    def _compute_silhouette_score(self, X: np.ndarray) -> float:
        """ """
        from sklearn.metrics import silhouette_score

        silhouette_avg = silhouette_score(X, self.labels)
        return silhouette_avg

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data points based on stored cluster means.

        Args:
            new_data (np.ndarray): The new data points for which predictions are to be made.

        Returns:
            np.ndarray: An array of cluster assignments for the new data points.
        """

        num_new_points = len(X)
        n_clusters = self.n_clusters

        cluster_assignments = np.zeros(num_new_points, dtype=int)

        for i in range(num_new_points):
            # Calculate distances between the new data point and cluster means
            distances = [np.linalg.norm(X[i] - self.cluster_centers[k]) for k in range(n_clusters)]
            cluster_assignments[i] = np.argmin(distances)
        return cluster_assignments
