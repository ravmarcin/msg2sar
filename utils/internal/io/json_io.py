import json


def open_json(path: str) -> dict:
    """
    JSON loading function

    Args:
        path (str): path to the JSON file 

    Returns:
        dict: loaded data
    """
    data = {}
    with open(path) as f:
        data = json.load(f)
    return data