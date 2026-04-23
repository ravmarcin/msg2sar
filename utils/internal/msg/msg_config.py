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


class MsgConfig:

    def __init__(self, config_path: str) -> None:
        self.config = open_json(config_path)
        self.__get_data()
        self.__get_full_paths()
        self.__load_bbox()
        self.__add_spatial_buffer()
        self.__load_aoi()
    
        log.info(
            f"MSG-CONFIG initialized with: \n "\
            f"AOI-{self.aoi_dict} \n"\
            f"DOWNLOAD_DIR-{self.download_dir} \n"\
            f"PROCESS_DIR-{self.process_dir} \n" \
            f"DEM_PATH-{self.dem_path} \n" \
        )

    def __get_data(self) -> None:
        self.job_name = self.config.get("job_name", "default")

        self.data = self.config.get("data", {})
        self.msg_processing = self.config.get("msg_processing", {})

        self.data_aoi = self.data.get("aoi", {})
        self.data_msg = self.data.get("msg", {})

        self.bursts = self.data_msg.get("bursts", [])
        self.reference = self.data_msg.get("ref_date", "")

        if self.reference: 
            self.reference = dt.strftime(dt.strptime(self.reference, '%Y%m%d'), '%Y-%m-%d')

    
    def __get_full_paths(self) -> None:
        if self.data_msg:
            self.data_dir = os.path.join(DATA_DIR, self.data_msg['data_child_dir'])
            self.download_dir = os.path.join(self.data_dir, self.data_msg['download_folder'])
            self.process_dir = os.path.join(self.data_dir, self.data_msg['process_folder'])
            self.output_dir = os.path.join(self.data_dir, self.data_msg['output_folder'])

            self.dem_path = os.path.join(self.download_dir, self.data_msg['dem_name'])

        else:
            self.data_dir = None
            self.download_dir = None
            self.process_dir = None
            self.output_dir = None
            self.dem_path = None
        
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
        self.bbox_buffer = buffer_bbox_wgs84(
            self.bbox,
            self.msg_processing.get("spatial_buffer_meters", 4000)
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

