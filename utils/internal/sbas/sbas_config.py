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

log = get_logger()


class SbasConfig:

    def __init__(self, config_path: str) -> None:
        self.config = open_json(config_path)
        self.__get_data()
        self.__get_full_paths()
        self.__load_aoi()
    
        log.info(
            f"SBAS-CONFIG initialized with: \n "\
            f"AOI-{self.aoi_dict} \n"\
            f"REFERENCE-{self.reference} \n"\
            f"DOWNLOAD_DIR-{self.download_dir} \n"\
            f"PROCESS_DIR-{self.process_dir} \n" \
            f"DEM_PATH-{self.dem_path} \n" \
        )

    def __get_data(self) -> None:
        self.job_name = self.config.get("job_name", "default")

        self.data = self.config.get("data", {})
        self.sar_processing = self.config.get("sar_processing", {})

        self.data_aoi = self.data.get("aoi", {})
        self.data_sar = self.data.get("sar", {})

        self.bursts = self.data_sar.get("bursts", [])
        self.reference = self.data_sar.get("ref_date", "")

        if self.reference: 
            self.reference = dt.strftime(dt.strptime(self.reference, '%Y%m%d'), '%Y-%m-%d')

    
    def __get_full_paths(self) -> None:
        if self.data_sar:
            self.data_dir = os.path.join(DATA_DIR, self.data_sar['data_child_dir'])
            self.download_dir = os.path.join(self.data_dir, self.data_sar['download_folder'])
            self.process_dir = os.path.join(self.data_dir, self.data_sar['process_folder'])
            self.output_dir = os.path.join(self.data_dir, self.data_sar['output_folder'])

            self.dem_path = os.path.join(self.download_dir, self.data_sar['dem_name'])
            self.landmask_path = os.path.join(self.process_dir, self.data_sar['landmask_name'])
            self.sbas_intf_path = os.path.join(self.process_dir, self.data_sar['sbas_intf_name'])
            self.sbas_corr_path = os.path.join(self.process_dir, self.data_sar['sbas_corr_name'])
            self.sbas_corr_stack_path = os.path.join(self.process_dir, self.data_sar['sbas_corr_stack_name'])
            self.sbas_uwrapped_path = os.path.join(self.process_dir, self.data_sar['sbas_uwrapped_name'])
            self.sbas_trend_path = os.path.join(self.process_dir, self.data_sar['sbas_trend_name'])
            self.sbas_disp_path = os.path.join(self.process_dir, self.data_sar['sbas_displacement_name'])
            self.sbas_uwrapped_impr_path = os.path.join(self.process_dir, self.data_sar['sbas_uwrapped_impr_name'])

        else:
            self.data_dir = None
            self.download_dir = None
            self.process_dir = None
            self.output_dir = None
            self.dem_path = None
            self.landmask_path = None
            self.sbas_intf_path = None
            self.sbas_corr_path = None
            self.sbas_corr_stack_path = None
            self.sbas_uwrapped_path = None
            self.sbas_trend_path = None
            self.sbas_disp_path = None
            self.sbas_uwrapped_impr_path = None
        
        if self.data_aoi:
            self.aoi_path = os.path.join(DATA_DIR, self.data_aoi['polygon_child_dir'])
        else:
            self.aoi_path = None


    def __load_aoi(self):
        if self.data_aoi and self.aoi_path:
            self.aoi_dict = get_aoi(json_path=self.aoi_path, aoi_name=self.data_aoi['name'])
            self.aoi = gpd.GeoDataFrame.from_features([self.aoi_dict])
            
        else:
            self.aoi_dict = {}
            self.aoi = None


