try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import xarray as xr
import numpy as np
import pandas as pd
from typing import Any
from utils.internal.log.logger import get_logger
from utils.internal.sbas.sbas_spec import SbasSpec
from utils.external.pygmtsar import utils as ut
from utils.external.pygmtsar.IO import IO
from utils.internal.dask.manager import DaskManager
from utils.internal.sbas.utils import shift_minor_modes


log = get_logger()


class SbasProcessor:


    def __init__(self, config_path: str) -> None:
        self.baseline_pairs = pd.DataFrame({})

        self.__default_values()
        self.spec = SbasSpec(config_path=config_path)
        self.config = self.spec.config
        self.dask_manager = DaskManager()
        self.sar_io = IO()


    def __default_values(self):
        self.geocoding_pixel_spacing: int = 1
        self.sbas_temporal_baseline_limit: int = 60
        self.sbas_spatial_wavelenght: int = 200
        self.sbas_goldstein_psize: int = 32
        self.ps_spatial_wavelenght: int = 100
        self.ps_landmask_correlation_threshold: float = 0.5
        self.ps_connected_components: int = 5
        self.sbas_pairs_covering_correlation_count: int = 3
        self.sbas_unwrapping_correlation_limit: float = 0.4
        self.trend_resolution: float = 15.0


    def save_nc_data(self, nc_data: xr.DataArray, path: str, engine: str = 'netcdf4') -> None:
        if self.dask_manager.running_client is not None:
            self.dask_manager.close_client()
        nc_data.to_netcdf(path, engine=engine)

    
    def load_nc_data(self, path: str, engine: str = 'netcdf4') -> xr.DataArray:
        return xr.open_dataarray(path, engine=engine)
    

    def save_cube_data(self, data: xr.DataArray, path: str) -> None:
        if self.dask_manager.running_client is None:
            self.dask_manager.start_client()

        self.sar_io.save_cube(data, path)

    
    def open_cube_data(self, path: str) -> xr.DataArray:

        try:
            if self.dask_manager.running_client is not None:
                self.dask_manager.close_client()
            self.dask_manager.close_client()
        except Exception as e:
            print(e)
    
        data = self.sar_io.open_cube(path)

        self.dask_manager.start_client()

        return data

    def run_process(self, process: callable, kwargs: dict = {}) -> Any:
        self.dask_manager.start_client()

        output = process(**kwargs)

        if self.dask_manager.running_client is not None:
            self.dask_manager.close_client()

        return output

    def reframe(self, n_jobs: int = 2) -> None:
        
        sbas_data = self.spec.get_stack(add_dem=False)
        
        log.info(f"Process start: REFRAMING")
        sbas_data.compute_reframe(self.config.aoi, n_jobs=n_jobs)
        log.info(f"Process end: REFRAMING")
        
        log.info('Cleaning')
        sbas_data = None


    def align(self, n_jobs: int = 2) -> None:
        
        sbas_data = self.spec.get_stack(add_dem=True)
        
        log.info(f"Process start: ALIGNMENT")
        sbas_data.compute_align(n_jobs=n_jobs)
        log.info(f"Process end: ALIGNMENT")
        
        log.info('Cleaning')
        sbas_data = None


    def geocoding(self, n_jobs: int = 1) -> None:
        
        sbas_data = self.spec.get_stack(add_dem=True)
        
        gpc = self.spec.spec.get("geocoding_pixel_spacing", self.geocoding_pixel_spacing)
    
        log.info(f"Process start: ALIGNMENT")
        sbas_data.compute_trans(coarsen=gpc, n_jobs=n_jobs)
        sbas_data.compute_trans_inv(coarsen=gpc)
        log.info(f"Process end: ALIGNMENT")
        
        log.info('Cleaning')
        sbas_data = None


    def multilook(self) -> None:
        # Computes look vectors: [longitude, latitude, elevation, look_E, look_N, look_U]

        sbas_data = self.spec.get_stack(add_dem=True)
    
        log.info(f"Process start: MULTILOOK")
        sbas_data.compute_satellite_look_vector()
        log.info(f"Process end: MULTILOOK")
        
        log.info('Cleaning')
        sbas_data = None

    
    def ps(self) -> None:
        
        sbas_data = self.spec.get_stack(add_dem=True)
    
        log.info(f"Process start: PS")
        sbas_data.compute_ps()
        log.info(f"Process end: PS")
        
        log.info('Cleaning')
        sbas_data = None


    def interferogram(self) -> None:
        
        sbas_data = self.spec.get_stack(add_dem=True)
    
        s_tbl = self.spec.spec.get("sbas_temporal_baseline_limit", self.sbas_temporal_baseline_limit)
        s_sw = self.spec.spec.get("sbas_spatial_wavelenght", self.sbas_spatial_wavelenght)
        s_gp = self.spec.spec.get("sbas_goldstein_psize", self.sbas_goldstein_psize)

        log.info(f"Process start: INTERFEROGRAM")
        self.baseline_pairs: pd.Dataframe = sbas_data.sbas_pairs(days=s_tbl)
        sbas_data.compute_interferogram_multilook(self.baseline_pairs, 'intf_mlook', wavelength=s_sw, psize=s_gp, weight=sbas_data.psfunction(), queue=4)
    
        log.info(f"Process end: INTERFEROGRAM")
        
        log.info('Cleaning')
        sbas_data = None


    def landmask(self) -> None:

        sbas_data = self.spec.get_stack(add_dem=True)

        ps_sw = self.spec.spec.get("ps_spatial_wavelenght", self.ps_spatial_wavelenght)
        ps_lct = self.spec.spec.get("ps_landmask_correlation_threshold", self.ps_landmask_correlation_threshold)
        ps_cc = self.spec.spec.get("ps_connected_components", self.ps_connected_components)
        landmask_path = self.config.landmask_path

        log.info(f"Process start: LANDMASK")

        log.info('Get PS mask')
        psmask_sbas = sbas_data.multilooking(sbas_data.psfunction(), coarsen=(1,4), wavelength=ps_sw) > ps_lct

        log.info('Get topography')
        topo_sbas = sbas_data.get_topo().interp_like(psmask_sbas, method='nearest')

        log.info('Generate Landmask')
        landmask_sbas = psmask_sbas & (np.isfinite(topo_sbas))
        landmask_sbas = ut.binary_opening(landmask_sbas, structure=np.ones((ps_cc, ps_cc)))
        landmask_sbas = np.isfinite(sbas_data.conncomp_main(landmask_sbas))
        landmask_sbas = ut.binary_closing(landmask_sbas, structure=np.ones((ps_cc, ps_cc)))
        landmask_sbas = np.isfinite(psmask_sbas.where(landmask_sbas))
        log.info(f"Process end: LANDMASK")

        log.info('Saving Landmask')
        self.save_cube_data(landmask_sbas, landmask_path)

        log.info('Cleaning')
        sbas_data = None
        landmask_sbas = None


    def sbas(self) -> None:

        sbas_data = self.spec.get_stack(add_dem=True)

        s_pccc = self.spec.spec.get("sbas_pairs_covering_correlation_count", self.sbas_pairs_covering_correlation_count)
        s_tbl = self.spec.spec.get("sbas_temporal_baseline_limit", self.sbas_temporal_baseline_limit)
        
        landmask_path = self.config.landmask_path
        sbas_intf_path = self.config.sbas_intf_path
        sbas_corr_path = self.config.sbas_corr_path
        sbas_corr_stack_path = self.config.sbas_corr_stack_path

        log.info(f"Process start: SBAS")
        ds_sbas = sbas_data.open_stack('intf_mlook')

        log.info('Loading Landmask')
        landmask_sbas = self.open_cube_data(landmask_path)

        log.info('Selecting best SBAS pairs')
        ds_sbas = ds_sbas.where(landmask_sbas)
        intf_sbas = ds_sbas.phase
        corr_sbas = ds_sbas.correlation
        # Add correlation
        baseline_pairs = sbas_data.sbas_pairs(days=s_tbl)
        baseline_pairs['corr'] = corr_sbas.mean(['y', 'x'])
        # Select best correlated pairs
        pairs_best = sbas_data.sbas_pairs_covering_correlation(baseline_pairs, count=s_pccc)
        intf_sbas = intf_sbas.sel(pair=pairs_best.pair.values)
        corr_sbas = corr_sbas.sel(pair=pairs_best.pair.values)

        log.info('Generate SBAS stack')
        corr_sbas_stack = corr_sbas.mean('pair')

        log.info(f"Process end: SBAS")

        log.info('Saving phase data')
        self.save_cube_data(intf_sbas, sbas_intf_path)

        log.info('Saving correlation data')
        self.save_cube_data(corr_sbas, sbas_corr_path)
        self.save_cube_data(corr_sbas_stack, sbas_corr_stack_path)

        log.info('Cleaning')
        sbas_data = None
        landmask_sbas = None
        intf_sbas = None
        corr_sbas = None
        corr_sbas_stack = None


    def unwrapping(self):

        sbas_data = self.spec.get_stack(add_dem=True)

        s_ucl = self.spec.spec.get("sbas_unwrapping_correlation_limit", self.sbas_unwrapping_correlation_limit)

        sbas_intf_path = self.config.sbas_intf_path
        sbas_corr_path = self.config.sbas_corr_path
        sbas_corr_stack_path = self.config.sbas_corr_stack_path
        sbas_uwrapped_path = self.config.sbas_uwrapped_path
        sbas_uwrapped_impr_path = self.config.sbas_uwrapped_impr_path

        log.info('Loading interferogram data')
        intf_sbas = self.open_cube_data(sbas_intf_path)

        log.info('Loading correlation data')
        corr_sbas = self.open_cube_data(sbas_corr_path)
        corr_sbas_stack = self.open_cube_data(sbas_corr_stack_path)

        log.info('Unwrapping')
        unwrap_sbas = sbas_data.unwrap_snaphu(
            intf_sbas.where(corr_sbas_stack > s_ucl),
            corr_sbas,
            conncomp=True
        )

        log.info('Saving uwrapped data')
        self.save_cube_data(unwrap_sbas, sbas_uwrapped_path)

        unwrap_sbas_c = unwrap_sbas.copy()

        for k in range(unwrap_sbas.phase.shape[0]):
            log.info(f"Improving {k + 1} uwrapped phase image")
            unwrap_sbas_c.phase[k] = shift_minor_modes(unwrap_sbas.phase[k])

        log.info('Saving improved uwrapped data')
        self.save_cube_data(unwrap_sbas_c, sbas_uwrapped_impr_path)

        log.info('Cleaning')
        sbas_data = None
        intf_sbas = None
        corr_sbas = None
        corr_sbas_stack = None
        unwrap_sbas = None
        unwrap_sbas_c = None


    def trend(self):

        sbas_data = self.spec.get_stack(add_dem=True)


        t_r = self.spec.spec.get("trend_resolution", self.trend_resolution)

        log.info('Loading data')
        sbas_corr_path = self.config.sbas_corr_path
        sbas_uwrapped_impr_path = self.config.sbas_uwrapped_impr_path
        sbas_trend_path = self.config.sbas_trend_path

        corr_sbas = self.open_cube_data(sbas_corr_path)
        unwrap_sbas_c = self.open_cube_data(sbas_uwrapped_impr_path)

        # Function to find the mode (most frequent value)
        # Solved by improved uwrapping!
        # unwrap_sbas_m = sbas_data.conncomp_main(unwrap_sbas_c, 1)

        # Trend estimation
        decimator_sbas = sbas_data.decimator(resolution=t_r, grid=(1,1))
        topo = decimator_sbas(sbas_data.get_topo())
        yy, xx = xr.broadcast(topo.y, topo.x)
        trend_sbas_m = sbas_data.regression(unwrap_sbas_c.phase,
                [
                    topo,
                    topo * yy,
                    topo * xx,
                    topo * yy * xx,
                    topo ** 2,
                    topo ** 2 * yy,
                    topo ** 2 * xx,
                    topo ** 2 * yy * xx,
                    yy,
                    xx,
                    yy * xx
                ], corr_sbas
        )
    
        # Add name
        trend_sbas_m.name = 'trend'

        log.info('Saving trend data')
        self.save_cube_data(trend_sbas_m, sbas_trend_path)

        log.info('Cleaning')
        sbas_data = None
        corr_sbas = None
        unwrap_sbas_c = None
        trend_sbas_m = None


    def displacement(self):

        sbas_data = self.spec.get_stack(add_dem=True)

        log.info('Loading data')
        sbas_corr_path = self.config.sbas_corr_path
        sbas_uwrapped_path = self.config.sbas_uwrapped_path
        sbas_trend_path = self.config.sbas_trend_path
        sbas_disp_path = self.config.sbas_disp_path

        corr_sbas = self.open_cube_data(sbas_corr_path)
        unwrap_sbas = self.open_cube_data(sbas_uwrapped_path)
        trend_sbas = self.open_cube_data(sbas_trend_path)

        # Function to find the mode (most frequent value)
        # Solved by improved uwrapping!
        # unwrap_sbas_m = sbas_data.conncomp_main(unwrap_sbas_c, 1)

        disp_sbas = sbas_data.los_displacement_mm(sbas_data.lstsq(unwrap_sbas.phase - trend_sbas, corr_sbas))

        log.info('Saving displacement data')
        self.save_cube_data(disp_sbas, sbas_disp_path)

        log.info('Cleaning')
        sbas_data = None
        corr_sbas = None
        unwrap_sbas = None
        disp_sbas = None


        