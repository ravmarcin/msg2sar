import platform, sys, os
PATH = os.environ['PATH']
from pygmtsar import __version__
print(__version__)

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from dask.distributed import Client
import dask
import warnings
warnings.filterwarnings('ignore')

# plotting modules
import pyvista as pv

from pygmtsar import S1, Stack, tqdm_dask, ASF, Tiles, XYZTiles, utils


from os.path import abspath, dirname, join
PROJ_PATH = dirname(abspath(abspath("")))
sys.path.insert(0, PROJ_PATH)

from settings.paths import setup
setup()


# The subswath is required for partial scene downloads and is not used for burst downloads.
# The orbit is used to define directory names.
ORBIT    = 'D' # descending (A for Ascending)
SUBSWATH = 3 # Id of a subswath



BURSTS = [
"S1_327247_IW3_20230420T044442_VV_8F60-BURST",
"S1_327247_IW3_20230408T044442_VV_C9C0-BURST",
"S1_327247_IW3_20230327T044442_VV_AAD6-BURST",
"S1_327247_IW3_20230315T044441_VV_147B-BURST",
"S1_327247_IW3_20230303T044442_VV_DCED-BURST",
"S1_327247_IW3_20230219T044442_VV_527E-BURST",
"S1_327247_IW3_20230207T044442_VV_FA04-BURST",
"S1_327247_IW3_20230126T044442_VV_C08C-BURST",
"S1_327247_IW3_20230114T044443_VV_1187-BURST",
"S1_327247_IW3_20230102T044443_VV_D42C-BURST"
]
BURSTS = [
    'S1_327247_IW3_20231216T044449_VV_1CC1-BURST',
    'S1_327247_IW3_20231204T044450_VV_42BA-BURST',
    'S1_327247_IW3_20231122T044450_VV_05AE-BURST',
    'S1_327247_IW3_20231110T044450_VV_B748-BURST',
    'S1_327247_IW3_20231017T044451_VV_49C0-BURST',
    'S1_327247_IW3_20231005T044451_VV_1C05-BURST',
    'S1_327247_IW3_20230923T044451_VV_3F01-BURST',
    'S1_327247_IW3_20230911T044450_VV_E969-BURST',
    'S1_327247_IW3_20230830T044450_VV_90C0-BURST',
    'S1_327247_IW3_20230818T044449_VV_C701-BURST',
    'S1_327247_IW3_20230806T044449_VV_FD68-BURST',
    'S1_327247_IW3_20230725T044448_VV_6BD6-BURST',
    'S1_327247_IW3_20230713T044447_VV_BDC7-BURST',
    'S1_327247_IW3_20230701T044446_VV_FCB0-BURST',
    'S1_327247_IW3_20230619T044446_VV_1703-BURST',
    'S1_327247_IW3_20230607T044445_VV_2A6C-BURST',
    'S1_327247_IW3_20230514T044444_VV_43BE-BURST',
    'S1_327247_IW3_20230526T044444_VV_8A01-BURST',
    'S1_327247_IW3_20230502T044443_VV_C50D-BURST'
]

print (f'Bursts defined: {len(BURSTS)}')

POLAR = 'VV'


from settings.paths import DATA_DIR
from utils.internal.io.json_io import open_json
from settings.paths import KEYS_DIR
from utils.internal.geo.aoi import get_aoi

main_folder = join(DATA_DIR, 'sar/sbas/desc/2023/bogo')

WORKDIR = os.path.join(main_folder, 'raw')
DATADIR = os.path.join(main_folder, 'data')



# define DEM filename inside data directory
DEM = f'{DATADIR}/dem.nc'
aoi_name = 'bogo'
aois_path = join(DATA_DIR, 'polygons/aoi.geojson')

AOI_dict = get_aoi(json_path=aois_path, aoi_name=aoi_name)
AOI = gpd.GeoDataFrame.from_features([AOI_dict])


secrets = open_json(join(KEYS_DIR, 'keys.json'))

# Set these variables to None and you will be prompted to enter your username and password below.
asf = ASF(secrets['asf']['username'], secrets['asf']['password'])



# Optimized scene downloading from ASF - only the required subswaths and polarizations.
# Subswaths are already encoded in burst identifiers and are only needed for scenes.
#print(asf.download(DATADIR, SCENES, SUBSWATH))

for burst in BURSTS:
    print(asf.download(DATADIR, [burst], skip_exist=True, polarization=POLAR, n_jobs=1))
    
    
    
# scan the data directory for SLC scenes and download missed orbits

orbits = S1.download_orbits(DATADIR, S1.scan_slc(DATADIR))


# download Copernicus Global DEM 1 arc-second

dem_geom = Tiles().download_dem(AOI, filename=DEM)