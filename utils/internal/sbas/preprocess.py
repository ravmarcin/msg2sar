import platform, sys, os
from os.path import abspath, dirname, join
PATH = os.environ['PATH']
from pygmtsar import __version__
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from dask.distributed import Client
import dask
import warnings
import subprocess
warnings.filterwarnings('ignore')

# plotting modules
import pyvista as pv
from pygmtsar import S1, Stack, tqdm_dask, ASF, Tiles, XYZTiles, utils

FILE_DIR = dirname(abspath(__file__))
PROJ_PATH = dirname(dirname(FILE_DIR))
sys.path.insert(0, PROJ_PATH)

from settings.paths import setup
setup()

from settings.paths import DATA_DIR
from utils.internal.io.json_io import open_json
from settings.paths import KEYS_DIR
from utils.internal.io.s1_stack import init_stack
from utils.internal.geo.aoi import get_aoi
from utils.internal.log.logger import get_logger

log = get_logger()

main_folder = join(DATA_DIR, 'sar/sbas/desc/2023/bogo_pl_test')
# The subswath is required for partial scene downloads and is not used for burst downloads.
# The orbit is used to define directory names.
ORBIT    = 'D'
SUBSWATH = 3
REFERENCE = '2023-02-19'

aoi_name = 'bogo'
aois_path = join(DATA_DIR, 'polygons/aoi.geojson')

# define DEM filename inside data directory
DEM = f'{main_folder}/data/dem.nc'


def get_spec(
        work_dir: str,
        orbit: str,
        subswath,
        ref: str,
        aoi_name: str,
        aois_path: str,
        dem_path: str = None,
        raw_data_dir: str = None,
        process_data_dir: str = None
    ) -> dict:

    """
    Function to define the processing specification
    
    """

    # Load geojson with outline polygon
    aoi_dict: dict = get_aoi(json_path=aois_path, aoi_name=aoi_name)
    
    # Setup WORKDIR, DATADIR and DEM_PATH
    if raw_data_dir is None:
        WORKDIR: str = os.path.join(work_dir, 'raw')
    else:
        WORKDIR: str = raw_data_dir
        
    if process_data_dir is None:
        DATADIR: str = os.path.join(work_dir, 'data')
    else:
        DATADIR: str = process_data_dir
        
    if dem_path is None:
        DEM_PATH: str = os.path.join(DATADIR, 'dem.nc')
    else:
        DEM_PATH: str = dem_path
        
    # Setup processing spec
    spec: dict = dict(
        WORKDIR = WORKDIR,
        DATADIR = DATADIR,
        ORBIT = orbit,
        SUBSWATH = subswath,
        REF = ref,
        AOI = gpd.GeoDataFrame.from_features([aoi_dict]),
        DEM_PATH = DEM_PATH
    )
    
    return spec
    

def cleanup():
    pass
    #subprocess.run(['bash', f'.{os.path.join(FILE_DIR, 'cleanup.sh')}'], check=True, text=True)

class PreprocessSBAS:
    
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        
    def init_sbas(
            self,
            dem: bool = False,
            verbose: bool = True,
            drop_if_exists: bool = False
        ) -> Stack:
        dem_path: str = self.spec['DEM_PATH'] if dem else None
        sbas: Stack = init_stack(
            dem=dem_path,
            aoi=self.spec['AOI'],
            ref=self.spec['REF'],
            data_dir=self.spec['WORKDIR'],
            work_dir=self.spec['DATADIR'],
            verbose=verbose,
            drop_if_exists=drop_if_exists
        )
        log.info(
            f"SBAS initialized with: \n "\
            f"AOI-{self.spec['AOI']} \n"\
            f"REFERENCE-{self.spec['REF']} \n"\
            f"WORKDIR-{self.spec['WORKDIR']} \n"\
            f"DATADIR-{self.spec['DATADIR']} \n" \
            f"DEM_PATH-{dem_path} \n" 
        )
        return sbas

    def reframe(
            self,
            n_jobs: int = 2,
            verbose: bool = True,
            drop_if_exists: bool = False,
            process_name: str = 'REFRAMING'
        ) -> None:
        
        sbas = self.init_sbas(
            dem=False,
            verbose=verbose,
            drop_if_exists=drop_if_exists
        )
        
        log.info(f"Process start: {process_name}")
        sbas.compute_reframe(self.spec['AOI'], n_jobs=n_jobs)
        log.info(f"Process end: {process_name}")
        
        sbas = None
        cleanup()
        
    def align(
            self,
            n_jobs: int = 2,
            verbose: bool = True,
            drop_if_exists: bool = False,
            process_name: str = 'ALIGNMENT'
        ) -> None:
        
        sbas = self.init_sbas(
            dem=True,
            verbose=verbose,
            drop_if_exists=drop_if_exists
        )
        
        log.info(f"Process start: {process_name}")
        sbas.compute_align(n_jobs=n_jobs)
        log.info(f"Process end: {process_name}")
        
        sbas = None
        cleanup()
        
