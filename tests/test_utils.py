import sys

import numpy as np
import pytest

from backbone_learn.utils.utils import Utils


def test_keep_lowest():
    print(sys.path)
    # Test keeping 3 lowest values
    arr = np.array([5, 2, 7, 3, 8, 1])
    expected = np.array([2, 3, 1])
    assert np.array_equal(Utils.keep_lowest(arr, 3), expected)

    # Test keeping lowest value
    arr = np.array([5, 2, 7])
    expected = np.array([2])
    assert np.array_equal(Utils.keep_lowest(arr, 1), expected)

    # Test empty array
    arr = np.array([])
    expected = np.array([])
    assert np.array_equal(Utils.keep_lowest(arr, 0), expected)


def test_keep_highest_standard():
    arr = np.array([1, 3, 2, 4, 5])
    assert np.array_equal(Utils.find_idx_highest(arr, 3), np.array([1, 3, 4]))


def test_keep_highest_edge_cases():
    # Edge case: Empty array
    assert Utils.find_idx_highest(np.array([]), 0).size == 0

    # Edge case: num_keep is equal to the length of the array
    arr = np.array([1, 2, 3])
    assert np.array_equal(Utils.find_idx_highest(arr, 3), np.array([0, 1, 2]))


def test_keep_highest_error_handling():
    arr = np.array([1, 2, 3])

    # num_keep is negative
    with pytest.raises(ValueError):
        Utils.find_idx_highest(arr, -1)

    # num_keep is larger than array length
    with pytest.raises(ValueError):
        Utils.find_idx_highest(arr, 4)


def test_merge_lists_and_sort():
    # Test with a simple case
    input_lists = [[1, 3, 2], [3, 4], [5, 1]]
    expected_output = [1, 2, 3, 4, 5]
    assert Utils.merge_lists_and_sort(input_lists) == expected_output

    # Test with empty lists
    input_lists = [[], [1], [], [2, 1]]
    expected_output = [1, 2]
    assert Utils.merge_lists_and_sort(input_lists) == expected_output

    # Test with all empty lists
    input_lists = [[], []]
    expected_output = []
    assert Utils.merge_lists_and_sort(input_lists) == expected_output

    # Test with tuples
    input_lists = [[(1, 2), (3, 4)], [(3, 4), (5, 6)], [(1, 2)]]
    expected_output = [(1, 2), (3, 4), (5, 6)]
    assert Utils.merge_lists_and_sort(input_lists) == expected_output


def test_find_common_tuples():
    assert Utils.find_common_tuples([[(1, 2), (1, 3)], [(2, 3), (1, 3)], [(1, 3), (4, 5)]]) == [
        (1, 3)
    ], "Test with three sublists failed"
    assert (
        Utils.find_common_tuples([[(1, 2), (3, 4)], [(5, 6)], [(7, 8)]]) == []
    ), "Test with no common tuples failed"
    assert Utils.find_common_tuples([[(1, 2), (3, 4)], [(1, 2), (4, 5)], [(1, 2), (5, 6)]]) == [
        (1, 2)
    ], "Test with common tuple in all lists failed"
    assert Utils.find_common_tuples([]) == [], "Test with empty list failed"
    assert Utils.find_common_tuples([[]]) == [], "Test with single empty sublist failed"


def test_generate_index_pairs():
    # Test case with specific inputs
    total_points = 4
    excluded_pairs = [(0, 2), (1, 3)]

    # Expected output
    expected_output = [(0, 1), (0, 3), (1, 2), (2, 3)]

    # Asserting if the function output matches the expected output
    assert Utils.generate_index_pairs(total_points, excluded_pairs) == expected_output
