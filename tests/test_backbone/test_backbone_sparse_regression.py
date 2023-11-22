from sklearn.datasets import make_regression

from backbone_learn.backbone.backbone_sparse_regression import BackboneSparseRegression


def test_backbone_sparse_regression_initialization():
    backbone = BackboneSparseRegression(alpha=0.5, beta=0.3, num_subproblems=2)
    if not backbone.screen_selector.alpha == 0.5:
        raise AssertionError("Backbone screen_selector alpha not set correctly")
    if not backbone.beta == 0.3:
        raise AssertionError("Backbone beta not set correctly")
    if not backbone.num_subproblems == 2:
        raise AssertionError("Backbone num_subproblems not set correctly")


def test_backbone_sparse_regression_predict():
    X, y = make_regression(
        n_samples=100, n_features=50, n_informative=20, noise=0.1, random_state=42
    )
    backbone = BackboneSparseRegression(alpha=0.5, beta=0.3, num_subproblems=2)
    backbone.fit(X, y)
    predictions = backbone.predict(X)

    # Validate the predictions
    if not len(predictions) == len(y):
        raise AssertionError("Prediction length mismatch")


def test_backbone_sparse_regression_predict_no_screening():
    X, y = make_regression(
        n_samples=100, n_features=50, n_informative=20, noise=0.1, random_state=42
    )
    backbone = BackboneSparseRegression(alpha=0.5, beta=0.3, num_subproblems=2)
    backbone.screen_selector = None
    backbone.fit(X, y)
    predictions = backbone.predict(X)

    # Validate the predictions
    if not len(predictions) == len(y):
        raise AssertionError("Prediction length mismatch with no screening")


def test_backbone_sparse_regression_predict_no_backbone():
    X, y = make_regression(
        n_samples=100, n_features=50, n_informative=20, noise=0.1, random_state=42
    )
    backbone = BackboneSparseRegression(alpha=0.5, beta=0.3, num_subproblems=2)
    backbone.heuristic_solver = None
    backbone.fit(X, y)
    predictions = backbone.predict(X)

    # Validate the predictions
    if not len(predictions) == len(y):
        raise AssertionError("Prediction length mismatch with no backbone")
