import sys, os
import geopandas as gpd
from dask.distributed import Client
import warnings
warnings.filterwarnings('ignore')
from pygmtsar import S1, Stack

from os.path import abspath, dirname, join
PROJ_PATH = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PROJ_PATH)

from settings.paths import setup
setup()


from settings.paths import DATA_DIR
from utils.internal.io.json_io import open_json
from settings.paths import KEYS_DIR

main_folder = join(DATA_DIR, 'sar/sbas/desc/2023/bogo_pl_test')
WORKDIR = os.path.join(main_folder, 'raw')
DATADIR = os.path.join(main_folder, 'data')

# The subswath is required for partial scene downloads and is not used for burst downloads.
# The orbit is used to define directory names.
ORBIT    = 'D'
SUBSWATH = 3
REFERENCE = '2023-01-02'

AOIs = open_json(join(DATA_DIR, 'polygons/bogo.geojson'))
AOI = gpd.GeoDataFrame.from_features(AOIs['features'])

# define DEM filename inside data directory
DEM = f'{DATADIR}/dem.nc'

def pipe():
    print('Scene scan')
    scenes = S1.scan_slc(DATADIR)
    sbas = Stack(WORKDIR, drop_if_exists=True).set_scenes(scenes).set_reference(REFERENCE)

    print('Reframe')
    sbas.compute_reframe(AOI, n_jobs=1)
    
    print('DEM load')
    sbas.load_dem(DEM, AOI)

    print('Aligment')
    sbas.compute_align()
    
if __name__ == '__main__': 
    print('Setting up a client')
    # simple Dask initialization
    if 'client' in globals():
        client.close()
    client = Client(n_workers=2)

    pipe()
    
    # use the original Sentinel-1 resolution (1 pixel spacing)
