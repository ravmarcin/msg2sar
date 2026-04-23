try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

from utils.external.pygmtsar import Stack
from utils.internal.sbas.sbas_config import SbasConfig
from utils.internal.log.logger import get_logger
from utils.internal.io.s1_stack import init_stack


log = get_logger()


class SbasSpec:


    def __init__(self, config_path: str) -> None:
        self.config = SbasConfig(config_path=config_path)
        self.verbose = True
        self.drop_if_exists = False
        self.spec = self.config.sar_processing
        

    def get_stack(self, add_dem: bool = False) -> Stack:
        if not add_dem:
            stack = init_stack(
                dem=None,
                aoi=self.config.aoi,
                ref=self.config.reference,
                data_dir=self.config.download_dir,
                work_dir=self.config.process_dir,
                verbose=self.verbose,
                drop_if_exists=self.drop_if_exists
            )
            
        else:
            stack = init_stack(
                dem=self.config.dem_path,
                aoi=self.config.aoi,
                ref=self.config.reference,
                data_dir=self.config.download_dir,
                work_dir=self.config.process_dir,
                verbose=self.verbose,
                drop_if_exists=self.drop_if_exists
            )
        log.info(f"SBAS-STACK initialized")
        return stack
