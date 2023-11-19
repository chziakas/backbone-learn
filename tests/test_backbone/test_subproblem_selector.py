import numpy as np

from backbone_learn.backbone.subproblem_feature_selector import SubproblemFeatureSelector


def test_subproblem_feature_selector_initialization():
    utilities = np.array([0.1, 0.4, 0.6, 0.3])
    num_features_to_select = 2
    selector = SubproblemFeatureSelector(utilities, num_features_to_select)

    assert np.array_equal(selector.utilities, utilities)
    assert selector.num_features_to_select == num_features_to_select


def test_subproblem_feature_selector_selection():
    utilities = np.array([0.1, 0.4, 0.6, 0.3])
    num_features_to_select = 2
    selector = SubproblemFeatureSelector(utilities, num_features_to_select)

    selected_indices = selector.select()

    # Check if the length of the selected indices is correct
    assert len(selected_indices) == num_features_to_select

    # Check if selected indices are valid
    assert all([idx in range(len(utilities)) for idx in selected_indices])


def test_subproblem_feature_selector_probability_distribution():
    utilities = np.array([0, 10, 20, 30])
    num_features_to_select = 1
    selector = SubproblemFeatureSelector(utilities, num_features_to_select)

    selector.select()

    # In this case, the highest utility is significantly larger,
    # so it should be selected most of the time.
    # Run the selection multiple times to verify this.
    counts = np.zeros(len(utilities))
    for _ in range(1000):
        idx = selector.select()[0]
        counts[idx] += 1

    assert np.argmax(counts) == np.argmax(utilities)
