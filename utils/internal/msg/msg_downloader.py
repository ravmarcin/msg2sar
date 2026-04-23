try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

from datetime import datetime, timedelta
from math import prod
import os
import shutil
from utils.internal.msg.msg_config import MsgConfig
from utils.internal.io.json_io import open_json
from utils.internal.log.logger import get_logger
from utils.external.pygmtsar import Tiles
from settings.paths import KEYS_DIR
from eumdac.token import AccessToken
from eumdac.datastore import DataStore


log = get_logger()


class MsgDownloader:

    def __init__(self, config_path: str) -> None:
        self.config = MsgConfig(config_path=config_path)
        self.fmt = "%Y%m%d_%H%M%S"

        if self.config.data_msg:
            self.collection_id = self.config.data_msg.get("collection_id", "EO:EUM:DAT:0665")
            self.date_time = self.config.data_msg.get("date_time", [])
            self.temporal_buffer_min = self.config.msg_processing.get("temporal_buffer_min", 15)
            self.download_dir = self.config.download_dir
        self.__eumetsat_init()

    def __eumetsat_init(self):
        secrets = open_json(os.path.join(KEYS_DIR, 'keys.json'))
        key = secrets['eumetsat']['consumer_key']
        secret = secrets['eumetsat']['consumer_secret']
        token = AccessToken((key, secret))
        self.datastore = DataStore(token)

    def __get_start_end(self, date_time: str):
        central_time = datetime.strptime(date_time, self.fmt)
        start = central_time - timedelta(minutes=self.temporal_buffer_min)
        end = central_time + timedelta(minutes=self.temporal_buffer_min)
        return start, end
    
    def download_prod(self):
        collection = self.datastore.get_collection(self.collection_id)
        for date_time in self.date_time:

            data_time_dir = os.path.join(self.download_dir, date_time)
            os.makedirs(data_time_dir, exist_ok=True)

            start, end = self.__get_start_end(date_time)
            products = collection.search(dtstart=start, dtend=end)

            for product in products:
                product_id = str(product)
                log.info(f"Downloading {product_id}")
                with product.open() as f_src, open(os.path.join(data_time_dir, f_src.name), mode='wb') as f_dst:
                    shutil.copyfileobj(f_src, f_dst)

    def download_dem(self):
        # download ALOS DEM or Copernicus Global DEM 1 arc-second
        # Options: provider='GLO' (Copernicus), 'ALOS', or 'SRTM'
        self.dem = Tiles().download_dem(self.config.aoi, filename=self.config.dem_path, provider='GLO')



