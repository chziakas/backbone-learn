import time
from itertools import product

from l0bnb import gen_synthetic
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils import save_results

from backbone_learn.backbone.backbone_sparse_regression import BackboneSparseRegression
from backbone_learn.heuristic_solvers.lasso_regression import LassoRegression

# Define parameter ranges for Backbone parameters
alpha_range = [0.1, 0.5]
beta_range = [0.5, 0.9]
num_subproblems_range = [5]
num_iterations_range = [1]
# Define parameter ranges for Lobnb parameters
lambda_2_range = [0.001]
n_non_zeros = 4
max_nonzeros = 4

# Define range for features and other constants
n_features_range = [1000]
n_samples = 500
random_state = 17
time_limit = 1800
log_filename = "sparse_regression_results.json"
results = []


# Experiment loop
for n_features in n_features_range:
    # Generate synthetic regression data
    X, y, b = gen_synthetic(n=n_samples, p=n_features, supp_size=n_non_zeros)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Lasso regression model iteration for heuristic_model
    heuristic_model = LassoRegression()
    start_time = time.time()
    heuristic_model.fit(X_train, y_train)
    runtime = time.time() - start_time
    heuristic_model.keep_top_features(n_non_zeros)
    y_pred_heuristic = heuristic_model.predict(X_test)
    r2_score_heuristic = r2_score(y_test, y_pred_heuristic)

    # Record heuristic model results
    result_heuristic = {
        "model_name": "heuristic",
        "n_features": n_features,
        "R2 Score": r2_score_heuristic,
        "Runtime (seconds)": runtime,
    }
    results.append(result_heuristic)

    # BackboneSparseRegression model iterations for 'backbone' and 'exact' solvers
    for lambda_2 in lambda_2_range:
        # Exact msodel iteration using BackboneSparseRegression
        exact_model = BackboneSparseRegression(max_nonzeros=max_nonzeros, time_limit=time_limit)
        exact_model.screen_selector = None
        exact_model.heuristic_solver = None
        start_time = time.time()
        exact_model.fit(X_train, y_train)
        runtime = time.time() - start_time
        y_pred_exact = exact_model.predict(X_test)
        r2_score_exact = r2_score(y_test, y_pred_exact)

        # Record exact model results
        result_exact = {
            "model_name": "exact",
            "n_features": n_features,
            "lambda_2": lambda_2,
            "R2 Score": r2_score_exact,
            "Runtime (seconds)": runtime,
        }
        results.append(result_exact)

        # Backbone model iteration using BackboneSparseRegression
        for alpha, beta, num_subproblems, num_iterations in product(
            alpha_range, beta_range, num_subproblems_range, num_iterations_range
        ):
            backbone_model = BackboneSparseRegression(
                alpha=alpha,
                beta=beta,
                num_subproblems=num_subproblems,
                num_iterations=num_iterations,
                lambda_2=lambda_2,
                max_nonzeros=max_nonzeros,
                time_limit=time_limit,
            )
            start_time = time.time()
            backbone_model.fit(X_train, y_train)
            runtime = time.time() - start_time
            y_pred_backbone = backbone_model.predict(X_test)
            backbone_size = len(backbone_model.variables_exact_idx)
            r2_score_backbone = r2_score(y_test, y_pred_backbone)

            # Record backbone model results
            result_backbone = {
                "model_name": "backbone",
                "n_features": n_features,
                "backbone_size": backbone_size,
                "alpha": alpha,
                "beta": beta,
                "num_subproblems": num_subproblems,
                "num_iterations": num_iterations,
                "lambda_2": lambda_2,
                "R2 Score": r2_score_backbone,
                "Runtime (seconds)": runtime,
            }
            results.append(result_backbone)

# Print or further process results
save_results(results, log_filename)
# Print or further process results
for result in results:
    print(result)
