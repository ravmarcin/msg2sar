import platform, sys, os
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
warnings.filterwarnings('ignore')
import pickle
import xarray

# plotting modules
import pyvista as pv

from pygmtsar import S1, Stack, tqdm_dask, ASF, Tiles, XYZTiles, utils

from os.path import abspath, dirname, join
PROJ_PATH = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PROJ_PATH)

from settings.paths import setup
setup()


from settings.paths import DATA_DIR
from utils.internal.io.json_io import open_json
from settings.paths import KEYS_DIR
from utils.internal.io.s1_stack import init_stack
from utils.internal.geo.aoi import get_aoi


main_folder = join(DATA_DIR, 'sar/sbas/desc/2023/bogo_pl_test')
WORKDIR = os.path.join(main_folder, 'raw')
DATADIR = os.path.join(main_folder, 'data')

# The subswath is required for partial scene downloads and is not used for burst downloads.
# The orbit is used to define directory names.
ORBIT    = 'D'
SUBSWATH = 3
REFERENCE = '2023-02-19'

aoi_name = 'bogo'
aois_path = join(DATA_DIR, 'polygons/aoi.geojson')

AOI_dict = get_aoi(json_path=aois_path, aoi_name=aoi_name)
AOI = gpd.GeoDataFrame.from_features([AOI_dict])

# define DEM filename inside data directory
DEM = f'{DATADIR}/dem.nc'

max_baseline_days = 60

CORRLIMIT = 0.3

def pipe0(n_jobs=1):
    sbas = init_stack(
        dem=None,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    
def pipe1_1(n_jobs=1):
    sbas = init_stack(
        dem=None,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )

    print('Reframe')
    sbas.compute_reframe(AOI, n_jobs=n_jobs)
    
    sbas = None

def pipe1_2(n_jobs=1):
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    
    print('Aligment')
    sbas.compute_align(n_jobs=n_jobs)
    sbas = None
    
def pipe2(n_jobs=1):

    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    
    print('Geocoding')
    # use the original Sentinel-1 resolution (1 pixel spacing)
    pixel_spacing = 1
    
    sbas.compute_trans(coarsen=pixel_spacing, n_jobs=n_jobs)
    sbas.compute_trans_inv(coarsen=pixel_spacing)
    
    sbas = None

def pipe3():
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    
    print('multilook')
    sbas.compute_satellite_look_vector()
    
    sbas = None
    

def pipe4():
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    
    print('compute PS')
    sbas.compute_ps()
    
    sbas = None
    

def pipe6():
    
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
        
    baseline_pairs = sbas.sbas_pairs(days=max_baseline_days)
    
    sbas.compute_interferogram_multilook(baseline_pairs, 'intf_mlook', wavelength=200, psize=32, weight=sbas.psfunction(), queue=4)
    
    sbas = None
    
# Should be 7?  
def pipe7(dask_client):
    
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
        
    print('Generate Landmask')
    psmask_sbas = sbas.multilooking(sbas.psfunction(), coarsen=(1,4), wavelength=100) > 0.5
    topo_sbas = sbas.get_topo().interp_like(psmask_sbas, method='nearest')
    landmask_sbas = psmask_sbas&(np.isfinite(topo_sbas))
    landmask_sbas = utils.binary_opening(landmask_sbas, structure=np.ones((20,20)))
    landmask_sbas = np.isfinite(sbas.conncomp_main(landmask_sbas))
    landmask_sbas = utils.binary_closing(landmask_sbas, structure=np.ones((20,20)))
    landmask_sbas = np.isfinite(psmask_sbas.where(landmask_sbas))
    
    print('Closing dask')
    dask_client.close()
    
    print('Saving Landmask')
    landmask_sbas.to_netcdf(os.path.join(WORKDIR, "landmask.nc"), engine='netcdf4')
    
    sbas = None
    landmask_sbas = None
    #outdir = "/home/rav_marcin/projects/msg2sar/data/sar/sbas/desc/2023/bogo_pl_test/pickle"
    #outpath = os.path.join(outdir, 'landmask_sbas.pkl')
    #with open(outpath, 'wb') as f:  # open a text file
    #    pickle.dump(landmask_sbas, f) # serialize the list
    
    return True
    
    
def pipe8(dask_client):
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
        

    ds_sbas = sbas.open_stack('intf_mlook')
    
    # apply land mask
    print('Loading Landmask')
    landmask_sbas = xarray.open_dataarray(os.path.join(WORKDIR, "landmask.nc"), engine='netcdf4')
    
    # 
    print('Selecting best SBAS pairs')
    ds_sbas = ds_sbas.where(landmask_sbas)
    
    intf_sbas = ds_sbas.phase
    corr_sbas = ds_sbas.correlation
    
    # Add correlation
    baseline_pairs = sbas.sbas_pairs(days=max_baseline_days)
    baseline_pairs['corr'] = corr_sbas.mean(['y', 'x'])
    
    # Select best correlated pairs
    pairs_best = sbas.sbas_pairs_covering_correlation(baseline_pairs, 2)
    intf_sbas = intf_sbas.sel(pair=pairs_best.pair.values)
    corr_sbas = corr_sbas.sel(pair=pairs_best.pair.values)
    
    print('Generate SBAS stack')
    corr_sbas_stack = corr_sbas.mean('pair')
    corr_sbas_stack = sbas.sync_cube(corr_sbas_stack, 'corr_sbas_stack')
    
    print('Closing dask')
    dask_client.close()
    
    print('Saving generated data')
    intf_sbas.to_netcdf(os.path.join(WORKDIR, "phase_sbas.nc"), engine='netcdf4')
    corr_sbas.to_netcdf(os.path.join(WORKDIR, "corr_sbas.nc"), engine='netcdf4')
    corr_sbas_stack.to_netcdf(os.path.join(WORKDIR, "corr_sbas_stack.nc"), engine='netcdf4')
    
    sbas = None
    landmask_sbas = None
    intf_sbas = None
    corr_sbas = None
    corr_sbas_stack = None
    
    return True

def pipe9(dask_client):
    sbas = init_stack(
        dem=DEM,
        aoi=AOI,
        ref=REFERENCE,
        data_dir=DATADIR,
        work_dir=WORKDIR,
        verbose=True,
        drop_if_exists=False
    )
    

    print('Loading data')
    intf_sbas = xarray.open_dataarray(os.path.join(WORKDIR, "phase_sbas.nc"), engine='netcdf4')
    corr_sbas = xarray.open_dataarray(os.path.join(WORKDIR, "corr_sbas.nc"), engine='netcdf4')
    corr_sbas_stack = xarray.open_dataarray(os.path.join(WORKDIR, "corr_sbas_stack.nc"), engine='netcdf4')
    
    print('Unwrapping')
    unwrap_sbas = sbas.unwrap_snaphu(
        intf_sbas.where(corr_sbas_stack>CORRLIMIT),
        corr_sbas,
        conncomp=True
    )
    
    print('Closing dask')
    dask_client.close()
    
    print('Saving uwrapped array')
    unwrap_sbas.to_netcdf(os.path.join(WORKDIR, "unwrap_sbas.nc"), engine='netcdf4')
    
if __name__ == '__main__': 
    
    print('Setting up a client')
    # simple Dask initialization
    
    def start_client(n_workers=1, memory_limit="3.8GiB"):
        if 'dask_client' in globals():
            dask_client.close()
        
        dask_client = Client(n_workers=n_workers, memory_limit=memory_limit)
        return dask_client
    
    def _process(n_workers: int, process: callable, kwargs: dict = None):
        dask_client = start_client(n_workers)
        #try:
        if True:
            if isinstance(kwargs, dict) and 'dask_client' in list(kwargs.keys()):
                kwargs['dask_client'] = dask_client
                
            if kwargs is not None:
                output = process(**kwargs)
            else:
                output = process()
            if not output:
                print('Closing dask')
                dask_client.close()
        #except Exception as e:
        #    print(f'Error in {process.__name__}')
        #    print(e)
        #    print('Closing dask')
        #    dask_client.close()
    
    #_process(2, pipe1_1, dict(n_jobs=2))
    #_process(2, pipe1_2, dict(n_jobs=2))
    
    #_process(2, pipe2, dict(n_jobs=2))
    
    #_process(2, pipe3)
    
    #_process(1, pipe4)
    
    #_process(1, pipe6)
    
    #_process(1, pipe7, dict(dask_client=True))
    
    #_process(1, pipe8, dict(dask_client=True))
    
    
    #_process(2, pipe9, dict(dask_client=True))
    
    _process(1, pipe0)
    
