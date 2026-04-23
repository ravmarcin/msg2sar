try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import griddata, interp1d
from typing import Tuple, Optional
from utils.internal.log.logger import get_logger
from utils.internal.gnss.gnss_config import GnssConfig

log = get_logger()


class GnssProcessor:
    """
    GNSS data processor for atmospheric correction validation.

    Provides spatial and temporal interpolation of GNSS ZTD data
    to match InSAR geometry and timing.
    """

    def __init__(self, config_path: str):
        self.config = GnssConfig(config_path)
        self.gnss_data = None

    def load_gnss_data(self) -> pd.DataFrame:
        """
        Load GNSS data from CSV file.

        Returns:
            DataFrame with GNSS ZTD measurements
        """
        gnss_path = os.path.join(self.config.download_dir, self.config.output_name)

        if not os.path.exists(gnss_path):
            log.error(f"GNSS data file not found: {gnss_path}")
            return pd.DataFrame()

        self.gnss_data = pd.read_csv(gnss_path)
        self.gnss_data['datetime'] = pd.to_datetime(self.gnss_data['datetime'])

        log.info(
            f"Loaded GNSS data: {len(self.gnss_data)} measurements from "
            f"{self.gnss_data['station_id'].nunique()} stations"
        )

        return self.gnss_data

    def interpolate_ztd_spatial(
        self,
        target_lat: float,
        target_lon: float,
        timestamp: datetime,
        method: str = 'linear'
    ) -> Optional[float]:
        """
        Spatially interpolate ZTD to target location.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            timestamp: Target timestamp
            method: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            Interpolated ZTD value in meters
        """
        if self.gnss_data is None:
            self.load_gnss_data()

        if self.gnss_data.empty:
            return None

        # Filter for time window
        time_window = timedelta(hours=self.config.temporal_buffer_hours)
        time_mask = (self.gnss_data['datetime'] >= timestamp - time_window) & \
                   (self.gnss_data['datetime'] <= timestamp + time_window)

        df_time = self.gnss_data[time_mask]

        if df_time.empty:
            log.warning(f"No GNSS data found near timestamp {timestamp}")
            return None

        # Group by station and average over time window
        df_avg = df_time.groupby('station_id').agg({
            'lat': 'first',
            'lon': 'first',
            'ztd': 'mean'
        }).reset_index()

        if len(df_avg) < 3:
            log.warning(f"Insufficient GNSS stations ({len(df_avg)}) for spatial interpolation")
            # Return nearest station value
            return self._nearest_station_ztd(df_avg, target_lat, target_lon)

        # Perform spatial interpolation
        points = df_avg[['lon', 'lat']].values
        values = df_avg['ztd'].values
        target_point = np.array([[target_lon, target_lat]])

        try:
            interpolated = griddata(points, values, target_point, method=method)
            return float(interpolated[0])

        except Exception as e:
            log.error(f"Spatial interpolation failed: {e}")
            return self._nearest_station_ztd(df_avg, target_lat, target_lon)

    def _nearest_station_ztd(
        self,
        df: pd.DataFrame,
        target_lat: float,
        target_lon: float
    ) -> float:
        """
        Get ZTD from nearest station (fallback method).

        Args:
            df: DataFrame with station data
            target_lat: Target latitude
            target_lon: Target longitude

        Returns:
            ZTD from nearest station
        """
        distances = np.sqrt(
            (df['lat'] - target_lat)**2 +
            (df['lon'] - target_lon)**2
        )
        nearest_idx = distances.argmin()
        return float(df.iloc[nearest_idx]['ztd'])

    def interpolate_ztd_temporal(
        self,
        station_id: str,
        timestamps: list
    ) -> np.ndarray:
        """
        Temporally interpolate ZTD for a station at given timestamps.

        Args:
            station_id: GNSS station identifier
            timestamps: List of datetime objects

        Returns:
            Array of interpolated ZTD values
        """
        if self.gnss_data is None:
            self.load_gnss_data()

        # Filter for specific station
        station_data = self.gnss_data[
            self.gnss_data['station_id'] == station_id
        ].sort_values('datetime')

        if station_data.empty:
            log.error(f"No data found for station {station_id}")
            return np.full(len(timestamps), np.nan)

        # Convert to numeric for interpolation
        times_numeric = station_data['datetime'].astype(np.int64) // 10**9
        ztd_values = station_data['ztd'].values

        # Create interpolator
        interpolator = interp1d(
            times_numeric,
            ztd_values,
            kind='linear',
            fill_value='extrapolate'
        )

        # Interpolate at target times
        target_times_numeric = np.array([t.timestamp() for t in timestamps])
        interpolated_ztd = interpolator(target_times_numeric)

        return interpolated_ztd

    def compute_ztd_grid(
        self,
        sar_grid: xr.DataArray,
        timestamp: datetime,
        method: str = 'linear'
    ) -> xr.DataArray:
        """
        Compute ZTD on a SAR grid using spatial interpolation.

        Args:
            sar_grid: SAR data grid with lat/lon coordinates
            timestamp: Timestamp for ZTD computation
            method: Interpolation method

        Returns:
            DataArray with interpolated ZTD values on SAR grid
        """
        if self.gnss_data is None:
            self.load_gnss_data()

        if self.gnss_data.empty:
            log.error("No GNSS data available for grid computation")
            return None

        # Filter for time window
        time_window = timedelta(hours=self.config.temporal_buffer_hours)
        time_mask = (self.gnss_data['datetime'] >= timestamp - time_window) & \
                   (self.gnss_data['datetime'] <= timestamp + time_window)

        df_time = self.gnss_data[time_mask]

        if df_time.empty:
            log.warning(f"No GNSS data found near timestamp {timestamp}")
            return None

        # Average over time window per station
        df_avg = df_time.groupby('station_id').agg({
            'lat': 'first',
            'lon': 'first',
            'ztd': 'mean'
        }).reset_index()

        # Get SAR grid coordinates
        if 'lat' in sar_grid.coords and 'lon' in sar_grid.coords:
            lats = sar_grid.coords['lat'].values
            lons = sar_grid.coords['lon'].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            log.error("SAR grid missing lat/lon coordinates")
            return None

        # Prepare GNSS points and values
        points = df_avg[['lon', 'lat']].values
        values = df_avg['ztd'].values

        # Flatten grid for interpolation
        target_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

        # Perform interpolation
        try:
            interpolated_flat = griddata(
                points, values, target_points,
                method=method, fill_value=np.nan
            )
            interpolated_grid = interpolated_flat.reshape(lon_grid.shape)

            # Create xarray DataArray
            ztd_array = xr.DataArray(
                interpolated_grid,
                coords={'lat': lats, 'lon': lons},
                dims=['lat', 'lon'],
                name='ztd',
                attrs={
                    'units': 'meters',
                    'long_name': 'Zenith Tropospheric Delay',
                    'timestamp': str(timestamp),
                    'interpolation_method': method
                }
            )

            log.info(f"Computed ZTD grid: {interpolated_grid.shape}")
            return ztd_array

        except Exception as e:
            log.error(f"Grid interpolation failed: {e}")
            return None

    def save_ztd_grid(self, ztd_grid: xr.DataArray, filename: str) -> None:
        """
        Save ZTD grid to NetCDF file.

        Args:
            ztd_grid: DataArray with ZTD values
            filename: Output filename
        """
        output_path = os.path.join(self.config.download_dir, filename)
        ztd_grid.to_netcdf(output_path)
        log.info(f"ZTD grid saved to: {output_path}")
