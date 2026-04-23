import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage.interpolation import rotate
import geopandas as gpd
import math


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def find_rect_in_gdf(gdf: gpd.GeoDataFrame) -> np.ndarray:
    geom = gdf['geometry']
    intersect_p1 = geom.intersection(geom)
    geom_idx = geom.index.values.tolist()
    p = intersect_p1[geom_idx[0]]
    for i in range(1, len(geom.index)):
        p = intersect_p1[geom_idx[i]].intersection(p)
    x, y = p.exterior.coords.xy
    xy = np.array([x.tolist(), y.tolist()])
    return minimum_bounding_rectangle(xy.T)


def geojson_to_bbox(geojson_obj):
    """
    Convert a GeoJSON dict to a bounding box [minx, miny, maxx, maxy].
    """

    def extract_coords(geometry):
        """Recursively extract all coordinate pairs from geometry."""
        coords = []

        if geometry["type"] == "Point":
            coords.append(geometry["coordinates"])

        elif geometry["type"] in ("LineString", "MultiPoint"):
            coords.extend(geometry["coordinates"])

        elif geometry["type"] in ("Polygon", "MultiLineString"):
            for ring in geometry["coordinates"]:
                coords.extend(ring)

        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                for ring in polygon:
                    coords.extend(ring)

        elif geometry["type"] == "GeometryCollection":
            for geom in geometry["geometries"]:
                coords.extend(extract_coords(geom))

        return coords

    # Handle Feature / FeatureCollection
    geometries = []

    if geojson_obj["type"] == "Feature":
        geometries.append(geojson_obj["geometry"])

    elif geojson_obj["type"] == "FeatureCollection":
        for feature in geojson_obj["features"]:
            geometries.append(feature["geometry"])

    else:  # Raw geometry
        geometries.append(geojson_obj)

    all_coords = []
    for geom in geometries:
        if geom is not None:
            all_coords.extend(extract_coords(geom))

    if not all_coords:
        raise ValueError("No coordinates found in GeoJSON")

    xs = [coord[0] for coord in all_coords]
    ys = [coord[1] for coord in all_coords]

    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_to_geojson(bbox):
    """
    Convert a bounding box to a GeoJSON FeatureCollection.
    
    Parameters:
        bbox: [min_lon, min_lat, max_lon, max_lat]
    
    Returns:
        GeoJSON FeatureCollection with a Polygon feature representing the bbox
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Create polygon coordinates (rectangle with closing point)
    coordinates = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],  # Close the ring
    ]
    
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }
    
    return {
        "type": "FeatureCollection",
        "features": [feature]
    }


def buffer_bbox_wgs84(bbox, buffer_meters):
    """
    Adds a buffer (in meters) to a WGS84 bbox.

    Parameters:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        buffer_meters: buffer distance in meters

    Returns:
        Buffered bbox [min_lon, min_lat, max_lon, max_lat]
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    # Use center latitude for longitude scaling
    center_lat = (min_lat + max_lat) / 2.0

    # Constants
    meters_per_degree_lat = 111_320  # average, varies slightly
    meters_per_degree_lon = 111_320 * math.cos(math.radians(center_lat))

    # Convert meters to degrees
    delta_lat = buffer_meters / meters_per_degree_lat
    delta_lon = buffer_meters / meters_per_degree_lon

    return [
        min_lon - delta_lon,
        min_lat - delta_lat,
        max_lon + delta_lon,
        max_lat + delta_lat,
    ]