import time
from itertools import product

from sklearn.datasets import make_blobs
from utils import save_results

from backbone_learn.backbone.backbone_clustering import BackboneClustering
from backbone_learn.heuristic_solvers.kmeans_solver import KMeansSolver

# Define parameter ranges for BackboneClustering
beta_range = [1.0]
num_subproblems_range = [1, 2]
num_iterations_range = [1]
n_clusters_range = [2, 3]
n_features_range = [3, 20]
n_samples_range = [20]

# Constants
random_state = 17
time_limit = 3600
log_filename = "clustering_results.json"


results = []
# Experiment loop
for n_samples, n_clusters, n_features in product(
    n_samples_range, n_clusters_range, n_features_range
):
    # Generate synthetic data
    X, _ = make_blobs(
        n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state
    )

    # KMeansSolver model iteration (labeled as 'heuristic')
    heuristic_model = KMeansSolver(n_clusters=n_clusters)
    start_time = time.time()
    heuristic_model.fit(X)
    heuristic_runtime = time.time() - start_time
    heuristic_wcss = heuristic_model.wcss
    heuristic_silhouette = heuristic_model.silhouette_score

    # Record heuristic model results
    result_heuristic = {
        "model_name": "heuristic",
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_features": n_features,
        "WCSS": heuristic_wcss,
        "silhouette": heuristic_silhouette,
        "Runtime (seconds)": heuristic_runtime,
    }
    results.append(result_heuristic)

    # BackboneClustering model iterations for 'exact' solver
    exact_model = BackboneClustering(n_clusters=n_clusters, time_limit=time_limit)
    exact_model.screen_selector = None
    exact_model.heuristic_solver = None
    start_time = time.time()
    exact_model.fit(X)
    exact_runtime = time.time() - start_time
    exact_wcss = exact_model.exact_solver.wcss
    exact_silhouette = exact_model.exact_solver.silhouette_score

    # Record exact model results
    result_exact = {
        "model_name": "exact",
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_features": n_features,
        "WCSS": exact_wcss,
        "silhouette": exact_silhouette,
        "Runtime (seconds)": exact_runtime,
    }

    results.append(result_exact)

    # BackboneClustering model iterations for 'backbone' solvers
    for beta, num_subproblems, num_iterations in product(
        beta_range, num_subproblems_range, num_iterations_range
    ):
        backbone_model = BackboneClustering(
            beta=beta,
            num_subproblems=num_subproblems,
            num_iterations=num_iterations,
            n_clusters=n_clusters,
            time_limit=time_limit,
        )
        start_time = time.time()
        backbone_model.fit(X)
        backbone_runtime = time.time() - start_time
        backbone_wcss = backbone_model.exact_solver.wcss
        backbone_wcss_heur = backbone_model.heuristic_solver.wcss
        backbone_silhouette_heur = backbone_model.heuristic_solver.silhouette_score
        backbone_silhouette = backbone_model.exact_solver.silhouette_score

        # Record backbone model results
        result_backbone = {
            "model_name": "backbone",
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "beta": beta,
            "num_subproblems": num_subproblems,
            "num_iterations": num_iterations,
            "WCSS": backbone_wcss,
            "WCSS_heur": backbone_wcss_heur,
            "silhouette": backbone_silhouette,
            "silhouette_heur": backbone_silhouette_heur,
            "Runtime (seconds)": backbone_runtime,
        }
        results.append(result_backbone)

save_results(results, log_filename)
# Print or further process results
for result in results:
    print(result)
