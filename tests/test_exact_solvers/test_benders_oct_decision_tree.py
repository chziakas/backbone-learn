import numpy as np
from sklearn.datasets import make_classification
from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree

from backbone_learn.exact_solvers.benders_oct_decision_tree import BendersOCTDecisionTree

def test_preprocess_features():
    """Test the preprocessing of features."""
    model = BendersOCTDecisionTree(n_bins=3)
    X = np.array([[1, 2, 6], [4, 5, 9], [7, 8, 12]])
    X_preprocessed = model.preprocess_features(X)
    assert X_preprocessed.shape[1] == X.shape[1]*3
    assert np.all(X_preprocessed >= 0)

def test_fit_predict():
    """Test fitting and predicting with the model."""
    model = BendersOCTDecisionTree()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert np.all(np.isin(predictions, [0, 1]))
