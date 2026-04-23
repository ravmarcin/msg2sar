import numpy as np
import xarray as xr  # For NetCDF SEVIRI L2


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
from utils.internal.msg.msg_spec import MsgSpec
from utils.external.pygmtsar import utils as ut
from utils.external.pygmtsar.IO import IO
from utils.internal.dask.manager import DaskManager
from utils.internal.sbas.utils import shift_minor_modes


log = get_logger()


class SeviriProcessor:

    def __init__(self, config_path: str) -> None:
        self.config = MsgSpec(config_path=config_path)
        self.download_dir = self.config.download_dir


def build_Nwet_from_seviri(seviri_wv_file, seviri_bt_file):
    """
    Build N_wet(lat,lon,z) from SEVIRI TPW and BT.
    
    seviri_wv_file: NetCDF with 'TPW' (kg/m²)
    seviri_bt_file: 'BT_108' (K) for scale height
    Returns: N_wet_interp(lat,lon,z) function
    """
    ds_wv = xr.open_dataset(seviri_wv_file)
    TPW = ds_wv['TPW'].values  # (y,x)
    lons_wv, lats_wv = ds_wv['longitude'].values, ds_wv['latitude'].values
    
    ds_bt = xr.open_dataset(seviri_bt_file)
    BT_108 = ds_bt['BT_108'].values
    
    # Estimate scale height from BT gradient (proxy for stability)
    H_v = 2000.0 + 500.0 * (BT_108 - 273.15) / 20.0  # Rough: warmer → moister low levels
    H_v = np.clip(H_v, 1500, 3000)
    
    # Surface q_0 from TPW / integral
    rho0 = 1.2  # kg/m³ approx
    integ_factor = rho0 * H_v * (1 - np.exp(-12000/H_v))
    q0_map = TPW / integ_factor  # kg/kg
    
    def qv_interp(lat, lon, z):
        # 2D bilinear interp to (lat,lon)
        i_lat = np.searchsorted(lats_wv[:,0], lat) - 1
        i_lon = np.searchsorted(lons_wv[0,:], lon) - 1
        # Simple nearest for demo
        q0 = q0_map[min(i_lat,len(lats_wv)-2), min(i_lon,len(lons_wv)-2)]
        return q0 * np.exp(-z / H_v[min(i_lat,len(lats_wv)-2), min(i_lon,len(lons_wv)-2)])
    
    def N_wet_interp(lat, lon, z):
        """N_wet using Thayer formula."""
        qv = qv_interp(lat, lon, z)
        T = 288.15 - 0.0065 * z  # Std lapse rate
        e = qv * 1.2 * 287.0 * T / 0.622 / 100.0  # Pa approx
        k2, k3 = 22.8, 3.73  # C-band constants
        return (k2 * e / T + k3 * e / T**2)  # N-units
    return N_wet_interp


# Usage with previous LOS code (Sentinel-1 or MSG)
N_wet_seviri = build_Nwet_from_seviri('MSG_WV_YYYYMMDD.nc', 'MSG_BT_YYYYMMDD.nc')

# Then: L_w, phi_w = example_slant_delay(eof_path, ..., N_wet_seviri)
