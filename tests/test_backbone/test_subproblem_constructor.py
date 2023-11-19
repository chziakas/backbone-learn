import numpy as np

from backbone_learn.backbone.subproblem_costructor import SubproblemConstructor


def test_subproblem_constructor_initialization():
    utilities = np.array([1, 2, 3, 4, 5])
    beta = 0.5
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)

    assert constructor.num_features == 5
    assert constructor.beta == beta
    assert constructor.num_features_subproblem == 3  # 5 * 0.5 rounded up
    assert constructor.num_subproblems == num_subproblems


def test_subproblem_constructor_correct_number_of_subproblems():
    utilities = np.array([1, 2, 3, 4, 5])
    beta = 0.5
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    assert len(subproblems) == num_subproblems


def test_subproblem_constructor_correct_number_of_features_in_subproblems():
    utilities = np.array([1, 2, 3, 4, 5, 6])
    beta = 0.4
    num_subproblems = 2

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    for subproblem in subproblems:
        assert len(subproblem) == 3  # 6 * 0.4 rounded up


def test_subproblem_constructor_valid_indices():
    utilities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    beta = 0.3
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    total_features = len(utilities)

    for subproblem in subproblems:
        # Check if all indices are within range
        assert all(0 <= idx < total_features for idx in subproblem)

        # Check for duplicates within a subproblem
        assert len(set(subproblem)) == len(subproblem)


def test_create_subsets_from_X():
    # Simulating a dataset X
    np.random.seed(0)  # Setting a seed for reproducibility
    X = np.random.rand(100, 10)  # 100 samples, 10 features

    # Generate random utilities for features
    utilities = np.random.rand(10)

    # Initialize SubproblemConstructor
    beta = 0.3
    num_subproblems = 3
    constructor = SubproblemConstructor(utilities, beta, num_subproblems)

    # Create subproblems and corresponding subsets of X
    subproblems = constructor.construct_subproblems()
    subsets = [X[:, subproblem] for subproblem in subproblems]

    # Assertions and checks
    assert len(subsets) == num_subproblems
    for subset, subproblem in zip(subsets, subproblems):
        assert subset.shape[1] == len(subproblem)  # Check the number of features in each subset
