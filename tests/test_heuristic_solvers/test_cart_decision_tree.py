from sklearn.datasets import make_classification

from backbone_learn.heuristic_solvers.cart_decision_tree import CARTDecisionTree


def test_fit_method():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    cart = CARTDecisionTree()
    (X > 0.0).astype(int)
    cart.fit(X, y)

    if cart.model is None:
        raise AssertionError("CARTDecisionTree model not initialized after fit")
    if not isinstance(cart.auc_score, float):
        raise AssertionError("CARTDecisionTree auc_score is not a float value")
    if not (0 <= cart.auc_score <= 1):
        raise AssertionError("CARTDecisionTree auc_score is out of the expected range (0-1)")


def test_get_significant_features():
    # Create a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    (X > 0.0).astype(int)
    # Initialize and fit the CARTDecisionTree model
    cart = CARTDecisionTree()
    cart.fit(X, y)

    # Set a threshold for significant features
    threshold = 0.1
    significant_features = cart.get_relevant_variables(threshold)

    # Check if the method identifies significant features correctly
    if len(significant_features) < 0:
        raise AssertionError("Number of significant features is less than 0")
    if not all(cart.model.feature_importances_[idx] > threshold for idx in significant_features):
        raise AssertionError("Identified significant features do not meet the threshold")


def test_cart_decision_tree_predict():
    X_train, y_train = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42
    )
    X_test, _ = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)

    model = CARTDecisionTree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if len(predictions) != len(X_test):
        raise AssertionError("Number of predictions does not match number of test samples")
    if not all(pred in [0, 1] for pred in predictions):
        raise AssertionError("Predictions contain values outside of [0, 1]")
