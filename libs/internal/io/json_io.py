import os
import json
from shapely.geometry import shape
from shapely.wkt import dumps


def open_json(path: str) -> dict:
    """
    JSON loading function

    Args:
        path (str): path to the JSON file 

    Returns:
        dict: loaded data
    """
    data = {}

    if not os.path.exists(path):
        raise FileExistsError(f"{path} file does not exist")
    
    with open(path, "r") as f:
        data = json.load(f)

    return data


def geojson_to_wkt(geojson_path):
    """Convert GeoJSON Polygon to WKT for EUMDAC search."""
    with open(geojson_path) as f:
        gj = json.load(f)
    geom = shape(gj['features'][0]['geometry'])
    return dumps(geom)  # e.g. 'POLYGON((lon1 lat1, lon2 lat2, ...))'