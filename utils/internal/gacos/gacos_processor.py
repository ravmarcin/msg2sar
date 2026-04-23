try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import gzip
import requests
import numpy as np
import xarray as xr
import geopandas as gpd
from datetime import datetime
from typing import Optional, Tuple
from scipy.interpolate import RegularGridInterpolator
from utils.internal.log.logger import get_logger
from utils.internal.gacos.gacos_config import GacosConfig

log = get_logger()


class GacosProcessor:
    """
    GACOS atmospheric correction processor.

    Downloads and applies GACOS tropospheric correction products to
    InSAR interferograms for atmospheric phase screen removal.
    """

    def __init__(self, config_path: str):
        self.config = GacosConfig(config_path)
        self.gacos_resolution = 0.125  # GACOS native resolution in degrees

    def download_gacos(
        self,
        date: datetime,
        aoi: Optional[gpd.GeoDataFrame] = None
    ) -> Optional[str]:
        """
        Download GACOS .ztd file for given date and AOI.

        GACOS files are named: YYYYMMDD.ztd.gz

        Args:
            date: Date for GACOS data
            aoi: Area of interest (optional, uses config if not provided)

        Returns:
            Path to downloaded file, or None if download failed
        """
        if not self.config.enabled:
            log.info("GACOS processing is disabled in config")
            return None

        if aoi is None:
            aoi = self.config.aoi

        # Format date for GACOS filename
        date_str = date.strftime('%Y%m%d')
        filename = f"{date_str}.ztd.gz"
        url = f"{self.config.download_url}{filename}"

        output_path = os.path.join(self.config.download_dir, filename)

        # Check if already downloaded
        if os.path.exists(output_path):
            log.info(f"GACOS file already exists: {output_path}")
            return output_path

        # Download file
        try:
            log.info(f"Downloading GACOS data from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Save compressed file
            with open(output_path, 'wb') as f:
                f.write(response.content)

            log.info(f"Downloaded GACOS data to: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            log.error(f"Failed to download GACOS data: {e}")
            return None

    def load_gacos_file(self, filepath: str) -> Optional[xr.DataArray]:
        """
        Load GACOS .ztd.gz file into xarray DataArray.

        GACOS format:
        - ASCII grid format
        - Global coverage: -180 to 180 lon, -90 to 90 lat
        - Resolution: 0.125 degrees
        - Units: meters (zenith delay)

        Args:
            filepath: Path to GACOS .ztd.gz file

        Returns:
            DataArray with GACOS ZTD data
        """
        try:
            # Check if file is gzipped
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt') as f:
                    data = f.read()
            else:
                with open(filepath, 'r') as f:
                    data = f.read()

            # Parse ASCII grid
            # GACOS format: values separated by spaces, row by row
            lines = data.strip().split('\n')

            # Try to parse as space-separated values
            values = []
            for line in lines:
                if line.strip():
                    row_values = [float(x) for x in line.split()]
                    values.append(row_values)

            data_array = np.array(values, dtype=np.float32)

            # Create coordinate arrays (global grid at 0.125 deg resolution)
            nlat, nlon = data_array.shape
            lats = np.linspace(90, -90, nlat)
            lons = np.linspace(-180, 180, nlon)

            # Create xarray DataArray
            gacos_da = xr.DataArray(
                data_array,
                coords={'lat': lats, 'lon': lons},
                dims=['lat', 'lon'],
                name='ztd',
                attrs={
                    'units': 'meters',
                    'long_name': 'Zenith Tropospheric Delay',
                    'source': 'GACOS',
                    'resolution': f'{self.gacos_resolution} degrees'
                }
            )

            log.info(f"Loaded GACOS data: {data_array.shape}")
            return gacos_da

        except Exception as e:
            log.error(f"Failed to load GACOS file {filepath}: {e}")
            return None

    def resample_to_sar_grid(
        self,
        gacos_data: xr.DataArray,
        sar_grid: xr.DataArray
    ) -> xr.DataArray:
        """
        Resample GACOS data (0.125° grid) to SAR geometry using bilinear interpolation.

        Args:
            gacos_data: GACOS ZTD data on native grid
            sar_grid: SAR data grid with lat/lon coordinates

        Returns:
            GACOS data resampled to SAR grid
        """
        # Get SAR grid coordinates
        if 'lat' in sar_grid.coords and 'lon' in sar_grid.coords:
            target_lats = sar_grid.coords['lat'].values
            target_lons = sar_grid.coords['lon'].values
        elif 'latitude' in sar_grid.coords and 'longitude' in sar_grid.coords:
            target_lats = sar_grid.coords['latitude'].values
            target_lons = sar_grid.coords['longitude'].values
        else:
            log.error("SAR grid missing lat/lon coordinates")
            return None

        # Create 2D grids for target coordinates
        if target_lats.ndim == 1 and target_lons.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
        else:
            lat_grid = target_lats
            lon_grid = target_lons

        # Subset GACOS to region of interest (for efficiency)
        lat_min, lat_max = lat_grid.min(), lat_grid.max()
        lon_min, lon_max = lon_grid.min(), lon_grid.max()

        # Add buffer
        buffer = 1.0  # degrees
        gacos_subset = gacos_data.sel(
            lat=slice(lat_max + buffer, lat_min - buffer),
            lon=slice(lon_min - buffer, lon_max + buffer)
        )

        # Create interpolator
        points = (gacos_subset.coords['lat'].values, gacos_subset.coords['lon'].values)
        values = gacos_subset.values

        interpolator = RegularGridInterpolator(
            points, values,
            method=self.config.resampling_method,
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate to SAR grid
        target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        resampled_flat = interpolator(target_points)
        resampled = resampled_flat.reshape(lat_grid.shape)

        # Create DataArray with SAR grid coordinates
        resampled_da = xr.DataArray(
            resampled,
            coords=sar_grid.coords,
            dims=sar_grid.dims,
            name='ztd_resampled',
            attrs={
                'units': 'meters',
                'long_name': 'GACOS ZTD resampled to SAR grid',
                'resampling_method': self.config.resampling_method
            }
        )

        log.info(f"Resampled GACOS to SAR grid: {resampled.shape}")
        return resampled_da

    def apply_correction(
        self,
        interferogram: xr.DataArray,
        gacos_master: xr.DataArray,
        gacos_slave: xr.DataArray,
        los_inc_angle: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply GACOS correction to interferogram.

        Formula:
        phase_corrected = phase_raw - (gacos_master - gacos_slave) / cos(inc_angle)

        The division by cos(inc_angle) converts zenith delay to slant range delay.

        Args:
            interferogram: Unwrapped interferogram phase (radians)
            gacos_master: GACOS ZTD for master date (meters)
            gacos_slave: GACOS ZTD for slave date (meters)
            los_inc_angle: Line-of-sight incidence angle (radians)

        Returns:
            Atmospherically corrected interferogram
        """
        # Compute differential tropospheric delay
        gacos_diff = gacos_master - gacos_slave

        # Convert zenith delay to slant delay
        # ZTD / cos(inc) gives slant delay along LOS
        cos_inc = np.cos(los_inc_angle)
        slant_delay = gacos_diff / cos_inc

        # Convert delay (meters) to phase (radians)
        # phase = (4 * pi / wavelength) * delay
        # For Sentinel-1: wavelength = 0.055465763 m (C-band)
        wavelength = 0.055465763
        phase_delay = (4 * np.pi / wavelength) * slant_delay

        # Apply correction
        corrected = interferogram - phase_delay

        # Create output DataArray
        corrected_da = xr.DataArray(
            corrected,
            coords=interferogram.coords,
            dims=interferogram.dims,
            name='phase_corrected',
            attrs={
                'units': 'radians',
                'long_name': 'GACOS-corrected interferometric phase',
                'wavelength_m': wavelength
            }
        )

        log.info(
            f"Applied GACOS correction: "
            f"mean correction = {np.nanmean(phase_delay.values):.3f} rad, "
            f"std = {np.nanstd(phase_delay.values):.3f} rad"
        )

        return corrected_da

    def process_interferogram(
        self,
        interferogram: xr.DataArray,
        master_date: datetime,
        slave_date: datetime,
        los_inc_angle: xr.DataArray,
        output_filename: Optional[str] = None
    ) -> Optional[xr.DataArray]:
        """
        Complete GACOS correction pipeline for a single interferogram.

        Args:
            interferogram: Input interferogram
            master_date: Master acquisition date
            slave_date: Slave acquisition date
            los_inc_angle: LOS incidence angle
            output_filename: Optional output filename

        Returns:
            Corrected interferogram or None if processing failed
        """
        # Download GACOS for both dates
        gacos_master_file = self.download_gacos(master_date)
        gacos_slave_file = self.download_gacos(slave_date)

        if gacos_master_file is None or gacos_slave_file is None:
            log.error("Failed to download GACOS data")
            return None

        # Load GACOS files
        gacos_master = self.load_gacos_file(gacos_master_file)
        gacos_slave = self.load_gacos_file(gacos_slave_file)

        if gacos_master is None or gacos_slave is None:
            log.error("Failed to load GACOS data")
            return None

        # Resample to SAR grid
        gacos_master_resampled = self.resample_to_sar_grid(gacos_master, interferogram)
        gacos_slave_resampled = self.resample_to_sar_grid(gacos_slave, interferogram)

        # Apply correction
        corrected = self.apply_correction(
            interferogram,
            gacos_master_resampled,
            gacos_slave_resampled,
            los_inc_angle
        )

        # Save if output filename provided
        if output_filename:
            output_path = os.path.join(self.config.output_dir, output_filename)
            corrected.to_netcdf(output_path)
            log.info(f"Saved corrected interferogram to: {output_path}")

        return corrected
