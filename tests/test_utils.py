import sys

import numpy as np

from backbone_learn.utils.utils import Utils


def test_keep_lowest():
    print(sys.path)
    # Test keeping 3 lowest values
    arr = np.array([5, 2, 7, 3, 8, 1])
    expected = np.array([2, 3, 1])
    if not np.array_equal(Utils.keep_lowest(arr, 3), expected):
        raise AssertionError("Test keeping 3 lowest values failed")

    # Test keeping lowest value
    arr = np.array([5, 2, 7])
    expected = np.array([2])
    if not np.array_equal(Utils.keep_lowest(arr, 1), expected):
        raise AssertionError("Test keeping lowest value failed")

    # Test empty array
    arr = np.array([])
    expected = np.array([])
    if not np.array_equal(Utils.keep_lowest(arr, 0), expected):
        raise AssertionError("Test empty array failed")


def test_keep_highest_standard():
    arr = np.array([1, 3, 2, 4, 5])
    if not np.array_equal(Utils.find_idx_highest(arr, 3), np.array([1, 3, 4])):
        raise AssertionError("Test keep highest standard failed")


def test_keep_highest_edge_cases():
    # Edge case: Empty array
    if not (Utils.find_idx_highest(np.array([]), 0).size == 0):
        raise AssertionError("Edge case with empty array failed")

    # Edge case: num_keep is equal to the length of the array
    arr = np.array([1, 2, 3])
    if not np.array_equal(Utils.find_idx_highest(arr, 3), np.array([0, 1, 2])):
        raise AssertionError("Edge case with num_keep equal to array length failed")


def test_keep_highest_error_handling():
    arr = np.array([1, 2, 3])

    # num_keep is negative
    try:
        Utils.find_idx_highest(arr, -1)
        raise AssertionError("No error raised for negative num_keep")
    except ValueError:
        pass

    # num_keep is larger than array length
    try:
        Utils.find_idx_highest(arr, 4)
        raise AssertionError("No error raised for num_keep larger than array length")
    except ValueError:
        pass


def test_merge_lists_and_sort():
    # Test with a simple case
    input_lists = [[1, 3, 2], [3, 4], [5, 1]]
    expected_output = [1, 2, 3, 4, 5]
    if not Utils.merge_lists_and_sort(input_lists) == expected_output:
        raise AssertionError("Simple case merge and sort failed")

    # Test with empty lists
    input_lists = [[], [1], [], [2, 1]]
    expected_output = [1, 2]
    if not Utils.merge_lists_and_sort(input_lists) == expected_output:
        raise AssertionError("Test with empty lists failed")

    # Test with all empty lists
    input_lists = [[], []]
    expected_output = []
    if not Utils.merge_lists_and_sort(input_lists) == expected_output:
        raise AssertionError("Test with all empty lists failed")

    # Test with tuples
    input_lists = [[(1, 2), (3, 4)], [(3, 4), (5, 6)], [(1, 2)]]
    expected_output = [(1, 2), (3, 4), (5, 6)]
    if not Utils.merge_lists_and_sort(input_lists) == expected_output:
        raise AssertionError("Test with tuples failed")


def test_find_common_tuples():
    if not Utils.find_common_tuples([[(1, 2), (1, 3)], [(2, 3), (1, 3)], [(1, 3), (4, 5)]]) == [
        (1, 3)
    ]:
        raise AssertionError("Test with three sublists failed")

    if not (Utils.find_common_tuples([[(1, 2), (3, 4)], [(5, 6)], [(7, 8)]]) == []):
        raise AssertionError("Test with no common tuples failed")

    if not Utils.find_common_tuples([[(1, 2), (3, 4)], [(1, 2), (4, 5)], [(1, 2), (5, 6)]]) == [
        (1, 2)
    ]:
        raise AssertionError("Test with common tuple in all lists failed")

    if not Utils.find_common_tuples([]) == []:
        raise AssertionError("Test with empty list failed")

    if not Utils.find_common_tuples([[]]) == []:
        raise AssertionError("Test with single empty sublist failed")


def test_generate_index_pairs():
    total_points = 4
    excluded_pairs = [(0, 2), (1, 3)]
    expected_output = [(0, 1), (0, 3), (1, 2), (2, 3)]

    if not Utils.generate_index_pairs(total_points, excluded_pairs) == expected_output:
        raise AssertionError("Test for generate index pairs failed")
