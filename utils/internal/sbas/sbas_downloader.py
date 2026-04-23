try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
from utils.internal.io.json_io import open_json
from utils.internal.sbas.sbas_config import SbasConfig
from utils.internal.log.logger import get_logger
from utils.external.pygmtsar import S1, Tiles, ASF
from settings.paths import KEYS_DIR


log = get_logger()


class SbasDownloader:


    def __init__(self, config_path: str) -> None:
        self.config = SbasConfig(config_path=config_path)
        if self.config.data_sar:
            self.bursts = self.config.data_sar.get("bursts", [])
            self.polarization = self.config.data_sar.get("polarization", [])
            self.download_dir = self.config.download_dir
        self.__asf_init()


    def __asf_init(self):
        secrets = open_json(os.path.join(KEYS_DIR, 'keys.json'))
        token = secrets['asf']['token']
        self.asf = ASF(secrets['asf']['username'], secrets['asf']['password'], token=token)


    def download_bursts(self):
        for burst in self.bursts:
            log.info(self.asf.download(
                self.config.download_dir,
                [burst],
                skip_exist=True,
                polarization=self.polarization,
                n_jobs=1)
            )

    def download_orbits(self):
        # scan the data directory for SLC scenes and download missed orbits
        self.orbits = S1.download_orbits(
            self.config.download_dir, 
            S1.scan_slc(self.config.download_dir)
        )

    def download_dem(self):
        # download Copernicus Global DEM 1 arc-second
        self.dem = Tiles().download_dem(self.config.aoi, filename=self.config.dem_path)
   