import numpy as np


class Utils:
    @staticmethod
    def keep_lowest(arr: np.ndarray, num_keep: int) -> np.ndarray:
        """
        Keep the specified number of lowest values in a numpy array.

        This function identifies the `num_keep` lowest values in the provided numpy array `arr`
        and returns a new array containing only these values.

        Parameters:
        - arr (np.ndarray): The input numpy array from which the lowest values are to be selected.
        - num_keep (int): The number of lowest values to keep from the array.

        Returns:
        - np.ndarray: A numpy array containing the lowest `num_keep` values from the input array.

        Raises:
        - ValueError: If `num_keep` is larger than the size of `arr` or if `num_keep` is negative.

        """
        if not (0 <= num_keep <= len(arr)):
            raise ValueError(
                "num_keep must be non-negative and less than or equal to the length of arr"
            )

        indices_keep = np.argpartition(arr, num_keep)[:num_keep]
        mask = np.zeros(len(arr), dtype=bool)
        mask[indices_keep] = True
        return arr[mask]

    @staticmethod
    def find_idx_highest(arr: np.ndarray, num_keep: int) -> np.ndarray:
        """
        Keep the specified number of highest values in a numpy array.

        This function identifies the `num_keep` highest values in the provided numpy array `arr`
        and returns the indices of these values.

        Parameters:
        - arr (np.ndarray): The input numpy array from which the highest values are to be selected.
        - num_keep (int): The number of highest values whose indices are to be kept.

        Returns:
        - np.ndarray: An array of indices corresponding to the highest `num_keep` values in the input array.

        Raises:
        - ValueError: If `num_keep` is larger than the size of `arr` or if `num_keep` is negative.

        """
        if not (0 <= num_keep <= len(arr)):
            raise ValueError(
                "num_keep must be non-negative and less than or equal to the length of arr"
            )

        # np.argpartition is used to find the indices of the `num_keep` highest values
        # We use -num_keep to find the highest values (since argpartition sorts ascendingly)
        indices = np.argpartition(arr, -num_keep)[-num_keep:]

        # Sort the indices to get them in the order they appear in the original array
        return np.sort(indices)

    @staticmethod
    def merge_lists_and_sort(list_of_lists):
        """
        Merges a list of lists into a single list, removes duplicates, and sorts the list.

        Args:
            list_of_lists (list of list of int): The list of lists to merge.

        Returns:
            list: A sorted list with unique elements.
        """
        merged_list = list(set(item for sublist in list_of_lists for item in sublist))
        merged_list.sort()
        return merged_list

    @staticmethod
    def find_common_tuples(list_of_lists):
        """
        Find tuples that are common to all sublists within a given list of lists.

        Parameters:
        list_of_lists (list of list of tuples): A list containing sublists, where each sublist contains tuples.

        Returns:
        list: A list of tuples that are common to all sublists.
        """
        if not list_of_lists or not all(list_of_lists):
            return []
        # Find common tuples by intersecting all sublists
        common_tuples = set(list_of_lists[0])
        for sublist in list_of_lists[1:]:
            common_tuples.intersection_update(sublist)

        return list(common_tuples)
