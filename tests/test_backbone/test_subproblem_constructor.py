import numpy as np

from backbone_learn.backbone.subproblem_costructor import SubproblemConstructor


def test_subproblem_constructor_initialization():
    utilities = np.array([1, 2, 3, 4, 5])
    beta = 0.5
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)

    if constructor.num_features != 5:
        raise AssertionError("Constructor num_features not set correctly")
    if constructor.beta != beta:
        raise AssertionError("Constructor beta not set correctly")
    if constructor.num_features_subproblem != 3:  # 5 * 0.5 rounded up
        raise AssertionError("Constructor num_features_subproblem not set correctly")
    if constructor.num_subproblems != num_subproblems:
        raise AssertionError("Constructor num_subproblems not set correctly")


def test_subproblem_constructor_correct_number_of_subproblems():
    utilities = np.array([1, 2, 3, 4, 5])
    beta = 0.5
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    if len(subproblems) != num_subproblems:
        raise AssertionError("Incorrect number of subproblems created")


def test_subproblem_constructor_correct_number_of_features_in_subproblems():
    utilities = np.array([1, 2, 3, 4, 5, 6])
    beta = 0.4
    num_subproblems = 2

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    for subproblem in subproblems:
        if len(subproblem) != 3:  # 6 * 0.4 rounded up
            raise AssertionError("Incorrect number of features in a subproblem")


def test_subproblem_constructor_valid_indices():
    utilities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    beta = 0.3
    num_subproblems = 3

    constructor = SubproblemConstructor(utilities, beta, num_subproblems)
    subproblems = constructor.construct_subproblems()

    total_features = len(utilities)

    for subproblem in subproblems:
        # Check if all indices are within range
        if not all(0 <= idx < total_features for idx in subproblem):
            raise AssertionError("Invalid indices in subproblem")

        # Check for duplicates within a subproblem
        if len(set(subproblem)) != len(subproblem):
            raise AssertionError("Duplicates found in a subproblem")


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
    if len(subsets) != num_subproblems:
        raise AssertionError("Incorrect number of subsets created")
    for subset, subproblem in zip(subsets, subproblems):
        if subset.shape[1] != len(subproblem):  # Check the number of features in each subset
            raise AssertionError("Mismatch in number of features in a subset")
