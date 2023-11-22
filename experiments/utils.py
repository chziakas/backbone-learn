import json
import os


def save_results(results, filename):
    """
    Saves the provided results to a file in JSON format.

    This function takes a dictionary of results and a filename, then writes the results to the specified file in
    the same directory as the script. The results are stored in JSON format with an indentation of 4 spaces for readability.

    Args:
        results (dict): A dictionary containing the results data to be saved.
        filename (str): The name of the file where the results will be saved. The file will be created in the same
            directory as the script.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)


def load_results(filename):
    """
    Loads existing results from a JSON file.

    This function attempts to open and read the specified JSON file. If the file exists, it parses the JSON content into a Python object (typically a list or dictionary, depending on the JSON structure) and returns it. If the file does not exist, the function returns an empty list.

    Args:
        filename (str): The name of the file from which to load the results.

    Returns:
        list or dict: The contents of the JSON file parsed into a Python object. Returns an empty list if the file does not exist.

    Raises:
        json.JSONDecodeError: If the file contents are not valid JSON.
        IOError: If the file cannot be opened for reasons other than non-existence.
    """
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
