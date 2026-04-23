try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import geopandas as gpd
from datetime import datetime as dt
from settings.paths import DATA_DIR
from utils.internal.geo.aoi import get_aoi
from utils.internal.io.json_io import open_json
from utils.internal.log.logger import get_logger
from utils.internal.geo.poly import buffer_bbox_wgs84, geojson_to_bbox, bbox_to_geojson

log = get_logger()


class GnssConfig:
    """
    Configuration loader for GNSS data processing.

    Follows the MsgConfig pattern for consistency across the codebase.
    Loads GNSS-specific configuration including station selection,
    temporal/spatial buffers, and output paths.
    """

    def __init__(self, config_path: str) -> None:
        self.config = open_json(config_path)
        self.__get_data()
        self.__get_full_paths()
        self.__load_bbox()
        self.__add_spatial_buffer()
        self.__load_aoi()

        log.info(
            f"GNSS-CONFIG initialized with: \n "
            f"AOI-{self.aoi_dict} \n"
            f"GNSS_STATIONS-{self.gnss_stations} \n"
            f"TEMPORAL_BUFFER_HOURS-{self.temporal_buffer_hours} \n"
            f"SPATIAL_BUFFER_KM-{self.spatial_buffer_km} \n"
            f"DOWNLOAD_DIR-{self.download_dir} \n"
        )

    def __get_data(self) -> None:
        self.job_name = self.config.get("job_name", "default")

        self.data = self.config.get("data", {})
        self.data_aoi = self.data.get("aoi", {})
        self.data_gnss = self.data.get("gnss", {})

        # GNSS-specific parameters
        self.gnss_stations = self.data_gnss.get("stations", "auto")
        self.temporal_buffer_hours = self.data_gnss.get("temporal_buffer_hours", 6)
        self.spatial_buffer_km = self.data_gnss.get("spatial_buffer_km", 100)
        self.epos_api_url = self.data_gnss.get("epos_api_url", "https://tcs.ah-epos.eu/")

    def __get_full_paths(self) -> None:
        if self.data_gnss:
            self.data_dir = os.path.join(DATA_DIR, self.data_gnss['data_child_dir'])
            self.download_dir = os.path.join(self.data_dir, self.data_gnss['download_folder'])
            self.output_name = self.data_gnss.get('output_name', 'gnss_ztd.csv')

            # Create directories if they don't exist
            os.makedirs(self.download_dir, exist_ok=True)
        else:
            self.data_dir = None
            self.download_dir = None
            self.output_name = None

        if self.data_aoi:
            self.aoi_path = os.path.join(DATA_DIR, self.data_aoi['polygon_child_dir'])
        else:
            self.aoi_path = None

    def __load_bbox(self):
        if self.data_aoi and self.aoi_path:
            self.bbox = geojson_to_bbox(open_json(self.aoi_path))
        else:
            self.bbox = None

    def __add_spatial_buffer(self):
        # Convert km to meters for buffer_bbox_wgs84
        buffer_meters = self.spatial_buffer_km * 1000
        self.bbox_buffer = buffer_bbox_wgs84(
            self.bbox,
            buffer_meters
        )

    def __load_aoi(self):
        if self.data_aoi and self.aoi_path:
            data = open_json(self.aoi_path)
            geojson = bbox_to_geojson(bbox=self.bbox_buffer)
            geojson['name'] = data['name']
            geojson['crs'] = data['crs']
            for i, feature in enumerate(geojson['features']):
                feature['properties'] = data['features'][i]['properties']

            self.aoi_dict = get_aoi(aois_data=geojson, json_path=self.aoi_path, aoi_name=self.data_aoi['name'])
            self.aoi = gpd.GeoDataFrame.from_features([self.aoi_dict])

        else:
            self.aoi_dict = {}
            self.aoi = None
