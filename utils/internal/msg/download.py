#!/usr/bin/env python3
"""
EUMETSAT IRS + SEVIRI downloader using EUMDAC.
pip install eumdac shapely geojson
"""

import os
import json
from eumdac.token import AccessToken
from eumdac.datastore import DataStore
from utils.internal.io.json_io import geojson_to_wkt


def download_eumetsat_polygon(
        user,
        password,
        geojson_path,
        start_time,
        end_time,
        output_dir="./downloads",
        collections=None
    ):
    """
    Download MTG-IRS and MSG-SEVIRI data over polygon.
    
    user/password: EOP credentials
    geojson_path: e.g. {'type': 'FeatureCollection', 'features': [{'geometry': {'type': 'Polygon', ...}}]}
    start_time/end_time: 'YYYY-MM-DDTHH:MM:SSZ' UTC
    collections: list or None (auto: IRS L1/L2, SEVIRI L1.5)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate
    credentials = ClientCredentials(user, password)
    datastore = DataStore(credentials)
    
    # Polygon WKT
    wkt = geojson_to_wkt(geojson_path)
    
    # Auto-collections if None (MTG-IRS + MSG-SEVIRI)
    if collections is None:
        collections = [
            # MTG-IRS Level 1 radiance (your guide)
            "EO:EUM:DAT:MSG:IRS_L1",
            # MTG-IRS Level 2 profiles (T, q_v)
            "EO:EUM:DAT:MSG:IRS_L2",
            # MSG-SEVIRI L1.5 HRIT/NetCDF
            "EO:EUM:DAT:MSG:HRSEVIRI",
            # MSG-SEVIRI L2 WV/TPW
            "EO:EUM:DAT:MSG:WV_002_013"  # Adjust per exact product
        ]
    
    products = []
    for collection in collections:
        print(f"Searching {collection}...")
        
        # OpenSearch query by polygon + time
        search = datastore.get_search(
            collection=collection,
            start_time=start_time,
            completion_time=end_time,
            spatial_shape=wkt
        )
        
        # Fetch results (paginated)
        hits = list(search.get())
        print(f"Found {len(hits)} products for {collection}")
        products.extend(hits)
    
    if not products:
        print("No products found.")
        return
    
    # Download all (parallel optional)
    print(f"Downloading {len(products)} products...")
    for product in products:
        product_id = product['id']
        filename = f"{product_id}.zip"  # Or .nc
        download_path = os.path.join(output_dir, filename)
        
        if os.path.exists(download_path):
            print(f"Skipping {filename}")
            continue
            
        print(f"Downloading {product_id}...")
        datastore.download(product_id, download_path)
    
    print("Download complete!")

# ---------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    USER = "your_eop_username"
    PASSWORD = "your_eop_password"
    
    # Example Warsaw polygon GeoJSON (save as warsaw_polygon.geojson)
    geojson_ex = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [20.9, 52.1], [21.1, 52.1], [21.1, 52.3],
                    [20.9, 52.3], [20.9, 52.1]
                ]]
            }
        }]
    }
    with open("warsaw_polygon.geojson", "w") as f:
        json.dump(geojson_ex, f)
    
    # Time window (UTC)
    start = "2026-02-17T00:00:00Z"
    end = "2026-02-20T23:59:59Z"
    
    download_eumetsat_polygon(
        USER, PASSWORD,
        "warsaw_polygon.geojson",
        start, end,
        output_dir="./eumetsat_data"
    )
