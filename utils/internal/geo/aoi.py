from utils.internal.io.json_io import open_json


def get_aoi(
    aois_data: dict,
    aoi_name: str,
    json_path: str = None,
    aoi_name_key: str = 'aoiName'
) -> dict:
    
    if aois_data is None:
        aois_data = open_json(json_path)
        
    aois = aois_data['features']
    aoi = {}
    for f in aois:
        if f['properties'][aoi_name_key] == aoi_name:
            aoi = f
            break
    if not aoi:
        aoi_names = [a['properties'][aoi_name_key] for a in aois_data['features']]
        print(f"{aoi_name} does not exists in {json_path} (available: {', '.join(aoi_names)})")
    return aoi