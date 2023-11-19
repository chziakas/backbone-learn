import json
import os
import time
from itertools import product

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import save_results
from backbone_learn.backbone.backbone_decision_tree import BackboneDecisionTree

# Define parameter ranges for Backbone parameters
alpha_range = [0.1, 0.5]
beta_range = [0.5, 0.9]
num_subproblems_range = [10]
num_iterations_range = [1]
# Define parameter ranges for FlowOCT parameters
depth_range = [2]
_lambda_range = [0.0, 0.5]

# Define dataset parameters
n_informative = 2
n_bins = 2
n_features_range = [4]
n_samples = 100
n_classes = 2
random_state = 17
time_limit = 3600
log_filename = "decision_tree_results.json"
results = []

# Experiment loop
for n_features in n_features_range:
    # Generate synthetic classification data
    X, y = make_classification(n_samples=n_samples, n_informative=n_informative,
                               n_features=n_features, n_classes=n_classes, random_state=random_state)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    for depth, _lambda in product(depth_range, _lambda_range):
        # CARTDecisionTree model iteration for heuristic_model
        heuristic_model = DecisionTreeClassifier(max_depth=depth)
        start_time = time.time()
        heuristic_model.fit(X_train, y_train)
        runtime = time.time() - start_time
        y_pred_heuristic = heuristic_model.predict(X_test)
        auc_score_heuristic = roc_auc_score(y_test, y_pred_heuristic)

        # Record heuristic model results
        result_heuristic = {
            "model_name": "heuristic",
            "n_features": int(n_features * n_bins),
            "AUC Score": auc_score_heuristic,
            "Runtime (seconds)": runtime,
        }
        results.append(result_heuristic)

        # BackboneDecisionTree model iterations for 'exact' solution
        exact_model = BackboneDecisionTree(depth=depth, _lambda=_lambda, time_limit=time_limit, n_bins=n_bins)
        start_time = time.time()
        exact_model.fit(X_train, y_train)
        runtime = time.time() - start_time
        y_pred_exact = exact_model.predict(X_test)
        auc_score_exact = roc_auc_score(y_test, y_pred_exact)

        # Record exact model results
        result_exact = {
            "model_name": "exact",
            "n_features": int(n_features * n_bins),
            "n_informative": n_informative,
            "depth": depth,
            "_lambda": _lambda,
            "AUC Score": auc_score_exact,
            "Runtime (seconds)": runtime,
        }
        results.append(result_exact)

        # BackboneDecisionTree model iterations for 'backbone' solution
        for alpha, beta, num_subproblems, num_iterations in product(alpha_range, beta_range, num_subproblems_range, num_iterations_range):
            backbone_model = BackboneDecisionTree(
                alpha=alpha,
                beta=beta,
                num_subproblems=num_subproblems,
                num_iterations=num_iterations,
                depth=depth,
                time_limit=time_limit,
                threshold=0.001,
                n_bins=n_bins
            )
            start_time = time.time()
            backbone_model.fit(X_train, y_train)
            runtime = time.time() - start_time
            y_pred_backbone = backbone_model.predict(X_test)
            auc_score_backbone = roc_auc_score(y_test, y_pred_backbone)

            # Record backbone model results
             # Record backbone model results
            result_backbone = {
                "model_name": "backbone",
                "n_features": int(n_features * n_bins),
                "n_informative": n_informative,
                "alpha": alpha,
                "beta": beta,
                "num_subproblems": num_subproblems,
                "num_iterations": num_iterations,
                "depth": depth,
                "_lambda": _lambda,
                "AUC Score": auc_score_backbone,
                "Runtime (seconds)": runtime,
            }
            results.append(result_backbone)


save_results(results, log_filename)
# Print or further process results
for result in results:
    print(result)