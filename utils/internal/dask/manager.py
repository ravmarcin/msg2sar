from dask.distributed import Client
from typing import Optional
import asyncio
from utils.internal.log.logger import get_logger


log = get_logger()


class DaskManager:


    def __init__(self) -> None:
        self.running_client: Optional[Client] = None
        self.running_workers_n: int = 0
        self.memory_limit: str = "30GiB"


    def close_client(self) -> None:
        if self.running_client is not None:
            log.info(f'Closing DASK client')
            try:
                self.running_client.close()
                log.info(f'DASK client closed')
            except asyncio.exceptions.CancelledError:
                log.info(f'DASK client probably closed already')
            except Exception as e:
                log.error(e)
            self.running_client = None
            self.running_workers_n = 0
        else:
            log.info(f'DASK client already closed')


    def start_client(self, n_workers: int = 1) -> None:

        if self.running_client is not None:
            self.close_client()
        
        try:
            log.info(f'Starting DASK client with {n_workers} workers and {self.memory_limit} memory limit')
            self.running_client = Client(n_workers=n_workers, memory_limit=self.memory_limit)
            self.running_workers_n = n_workers
            log.info(f'DASK client is running')
        except RuntimeError as e:
            log.info(f'DASK client is already running')