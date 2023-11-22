from sklearn.datasets import make_classification

from backbone_learn.heuristic_solvers.cart_decision_tree import CARTDecisionTree


def test_fit_method():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    cart = CARTDecisionTree()
    (X > 0.0).astype(int)
    cart.fit(X, y)

    assert cart.model is not None
    assert isinstance(cart.auc_score, float)
    assert 0 <= cart.auc_score <= 1


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
    assert len(significant_features) >= 0
    assert all(cart.model.feature_importances_[idx] > threshold for idx in significant_features)


def test_cart_decision_tree_predict():
    X_train, y_train = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42
    )
    X_test, _ = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)

    model = CARTDecisionTree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)
