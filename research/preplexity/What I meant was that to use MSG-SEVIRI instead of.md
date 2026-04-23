<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# What I meant was that to use MSG-SEVIRI instead of IRS to compute slant delays

MSG-SEVIRI water vapour and temperature products can approximate tropospheric refractivity for InSAR slant delays, but with **lower vertical resolution** than IRS (2D columnar/total instead of 3D profiles).[^1][^2]

## 1. SEVIRI products for $N_w$

- **WV 6.2/7.3 µm channels**: Columnar water vapour (total precipitable water, TPW in kg/m²).
- **IR 10.8/12 µm**: Cloud‑free brightness temperature → surface/upper air T.
- Level‑2 NWP‑assisted retrievals give pseudo‑profiles (e.g. 1000-300 hPa).[^3][^1]

Convert to wet refractivity $N_w(x,y)$:

$$
\text{TPW}(x,y) = \int_0^\infty q_v(z,x,y) \rho(z,x,y) \, dz
$$

Assume a scale height model:

$$
q_v(z,x,y) = q_0(x,y) \exp(-z/H_v(x,y))
$$

where $H_v \approx 2$ km from SEVIRI T gradient or climatology, and $q_0$ from TPW inversion.[^3]

## 2. Modified Python: SEVIRI → slant delay

Replace the `N_wet_interp` with a SEVIRI‑derived 2D field. Load via `satpy` or `xarray`.

```python
import numpy as np
import xarray as xr  # For NetCDF SEVIRI L2

def build_Nwet_from_seviri(seviri_wv_file, seviri_bt_file, z_grid=np.arange(0,12000,100)):
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
```


## 3. Discrete sum unchanged

Use the **exact same** `los_sampling_points_geo` (for MSG LOS) or Sentinel‑1 version, and `slant_delay_from_refractivity(pts, ds, N_wet_seviri)`. The discretisation is identical; only the $N_w$ source changes to SEVIRI columnar estimates.[^4][^1]

**Limitations**: SEVIRI TPW assumes exponential profile (coarse vertically). Better: assimilate with NWP for $H_v$. For precise InSAR, IRS 3D >> SEVIRI 2D, but SEVIRI has 15‑min cadence advantage.[^2][^3]

<div align="center">⁂</div>

[^1]: https://data.destination-earth.eu/data-portfolio/EO.EUM.DAT.MSG.HRSEVIRI

[^2]: https://wis2-gdc.weather.gc.ca/collections/wis2-discovery-metadata/items/urn:wmo:md:int-eumetsat:EO:EUM:DAT:MSG:HRSEVIRI?f=html

[^3]: https://www.cmsaf.eu/SharedDocs/Literatur/document/2013/saf_cm_dwd_atbd_sev_cld_1_1_pdf.pdf?__blob=publicationFile

[^4]: https://dl.iafastro.directory/event/IAC-2022/paper/69040/

