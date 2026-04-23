try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import geopandas as gpd
from settings.paths import DATA_DIR
from utils.internal.geo.aoi import get_aoi
from utils.internal.io.json_io import open_json
from utils.internal.log.logger import get_logger
from utils.internal.geo.poly import buffer_bbox_wgs84, geojson_to_bbox, bbox_to_geojson

log = get_logger()


class GacosConfig:
    """
    Configuration loader for GACOS atmospheric correction.

    Follows the MsgConfig pattern for consistency across the codebase.
    Loads GACOS-specific configuration including download URLs,
    output paths, and processing options.
    """

    def __init__(self, config_path: str) -> None:
        self.config = open_json(config_path)
        self.__get_data()
        self.__get_full_paths()
        self.__load_bbox()
        self.__load_aoi()

        log.info(
            f"GACOS-CONFIG initialized with: \n "
            f"AOI-{self.aoi_dict} \n"
            f"ENABLED-{self.enabled} \n"
            f"DOWNLOAD_DIR-{self.download_dir} \n"
            f"OUTPUT_DIR-{self.output_dir} \n"
        )

    def __get_data(self) -> None:
        self.job_name = self.config.get("job_name", "default")

        self.data = self.config.get("data", {})
        self.data_aoi = self.data.get("aoi", {})
        self.data_gacos = self.data.get("gacos", {})

        # GACOS-specific parameters
        self.enabled = self.data_gacos.get("enabled", True)
        self.download_url = self.data_gacos.get(
            "download_url",
            "http://www.gacos.net/ztd/"
        )
        self.resampling_method = self.data_gacos.get("resampling_method", "bilinear")
        self.apply_before_training = self.data_gacos.get("apply_before_training", True)

    def __get_full_paths(self) -> None:
        if self.data_gacos:
            self.data_dir = os.path.join(DATA_DIR, self.data_gacos['data_child_dir'])
            self.download_dir = os.path.join(self.data_dir, self.data_gacos['download_folder'])
            self.output_dir = os.path.join(self.data_dir, self.data_gacos['output_folder'])

            # Create directories if they don't exist
            os.makedirs(self.download_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.data_dir = None
            self.download_dir = None
            self.output_dir = None

        if self.data_aoi:
            self.aoi_path = os.path.join(DATA_DIR, self.data_aoi['polygon_child_dir'])
        else:
            self.aoi_path = None

    def __load_bbox(self):
        if self.data_aoi and self.aoi_path:
            self.bbox = geojson_to_bbox(open_json(self.aoi_path))
        else:
            self.bbox = None

    def __load_aoi(self):
        if self.data_aoi and self.aoi_path:
            data = open_json(self.aoi_path)
            geojson = bbox_to_geojson(bbox=self.bbox)
            geojson['name'] = data['name']
            geojson['crs'] = data['crs']
            for i, feature in enumerate(geojson['features']):
                feature['properties'] = data['features'][i]['properties']

            self.aoi_dict = get_aoi(aois_data=geojson, json_path=self.aoi_path, aoi_name=self.data_aoi['name'])
            self.aoi = gpd.GeoDataFrame.from_features([self.aoi_dict])

        else:
            self.aoi_dict = {}
            self.aoi = None
