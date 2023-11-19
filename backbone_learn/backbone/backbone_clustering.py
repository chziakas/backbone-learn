from ..exact_solvers.mio_clustering import MIOClustering
from ..heuristic_solvers.kmeans_solver import KMeansSolver
from .backbone_unsupervised import BackboneUnsupervised


class BackboneClustering(BackboneUnsupervised):
    """
    Specific implementation of the Backbone method for clustering.

    This class uses K-means for heuristic solving and retains MIO optimzer for exact solving.
    No screen selector is used in this approach,  as K-means is considered efficient for feature selection.

    Inherits from:
        BackboneBase (ABC): The abstract base class for backbone algorithms.
    """

    def set_solvers(self, n_clusters: int = 10, time_limit: int = 1000):
        """
        Initializes the clustering method with specified components.

        Args:
            n_clusters (int, optional): Number of clusters for K-means. Defaults to 10.
            time_limit (int): Time limit for the optimization process.
        """
        print(f"n_clusters_backbone:{n_clusters}")
        self.screen_selector = None  # No screen selector for this clustering approach
        self.heuristic_solver = KMeansSolver(n_clusters=n_clusters)
        self.exact_solver = MIOClustering(n_clusters=n_clusters, time_limit=time_limit)
