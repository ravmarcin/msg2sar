try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

from utils.internal.log.logger import get_logger
from utils.internal.msg.msg_config import MsgConfig
from utils.internal.msg.pymsg.stack_base import MsgStackBase


log = get_logger()


class MsgSpec:

    def __init__(self, config_path: str) -> None:
        self.config = MsgConfig(config_path=config_path)
        self.verbose = True
        self.drop_if_exists = False
        self.spec = self.config.msg_processing

    def get_spec(self):
        stack = MsgStackBase(
            data_dir=self.config.download_dir,
            work_dir=self.config.process_dir
        )
        log.info(f"MSG-SPEC initialized with: \n {self.spec}")
        return stack