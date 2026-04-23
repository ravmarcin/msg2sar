try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from typing import Tuple, List
from utils.internal.log.logger import get_logger
from utils.internal.gnss.gnss_config import GnssConfig

log = get_logger()


class GnssDownloader:
    """
    GNSS data downloader for EPOS Tropospheric Correction Service.

    Downloads zenith tropospheric delay (ZTD) data from EPOS for validation
    of atmospheric corrections in InSAR processing.
    """

    def __init__(self, config_path: str):
        self.config = GnssConfig(config_path)

    def download_gnss_stations(
        self,
        aoi: gpd.GeoDataFrame,
        date_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """
        Query EPOS API for GNSS stations within AOI and date range.

        Args:
            aoi: GeoDataFrame defining the area of interest
            date_range: Tuple of (start_date, end_date)

        Returns:
            DataFrame with columns: station_id, lat, lon, ztd, datetime
        """
        start_date, end_date = date_range

        # Add temporal buffer
        start_buffered = start_date - timedelta(hours=self.config.temporal_buffer_hours)
        end_buffered = end_date + timedelta(hours=self.config.temporal_buffer_hours)

        # Get bounding box from AOI
        bounds = aoi.total_bounds  # minx, miny, maxx, maxy

        log.info(
            f"Querying EPOS for GNSS stations:\n"
            f"  Date range: {start_buffered} to {end_buffered}\n"
            f"  Spatial bounds: {bounds}"
        )

        # Query EPOS API
        stations_data = self._query_epos_api(
            bounds=bounds,
            start_date=start_buffered,
            end_date=end_buffered
        )

        if stations_data.empty:
            log.warning("No GNSS stations found in the specified AOI and date range")
            return pd.DataFrame()

        # Save to CSV
        output_path = os.path.join(self.config.download_dir, self.config.output_name)
        stations_data.to_csv(output_path, index=False)
        log.info(f"GNSS data saved to: {output_path}")

        return stations_data

    def _query_epos_api(
        self,
        bounds: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Query EPOS Tropospheric Correction Service API.

        Args:
            bounds: [minx, miny, maxx, maxy] in WGS84
            start_date: Start datetime
            end_date: End datetime

        Returns:
            DataFrame with GNSS ZTD data
        """
        # EPOS API endpoint for tropospheric data
        api_url = f"{self.config.epos_api_url}api/v1/execute"

        # Build query parameters
        params = {
            'bbox': f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
            'starttime': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'format': 'json'
        }

        try:
            log.info(f"Sending request to EPOS API: {api_url}")
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            # Convert to DataFrame
            # Note: Actual EPOS response structure may differ
            # This is a template that needs to be adjusted based on real API response
            records = []
            for feature in data.get('features', []):
                props = feature.get('properties', {})
                coords = feature.get('geometry', {}).get('coordinates', [])

                if len(coords) >= 2:
                    records.append({
                        'station_id': props.get('station_id', 'unknown'),
                        'lat': coords[1],
                        'lon': coords[0],
                        'ztd': props.get('ztd', None),  # Zenith Tropospheric Delay in meters
                        'datetime': pd.to_datetime(props.get('datetime'))
                    })

            df = pd.DataFrame(records)
            log.info(f"Retrieved {len(df)} GNSS measurements from {df['station_id'].nunique()} stations")

            return df

        except requests.exceptions.RequestException as e:
            log.error(f"EPOS API request failed: {e}")
            return pd.DataFrame()

        except (KeyError, ValueError) as e:
            log.error(f"Failed to parse EPOS API response: {e}")
            return pd.DataFrame()

    def compute_reference_ztd(
        self,
        sar_timestamp: datetime,
        reference_point: Tuple[float, float] = None
    ) -> float:
        """
        Compute interpolated ZTD at reference point for InSAR correction.

        Args:
            sar_timestamp: SAR acquisition timestamp
            reference_point: (lat, lon) tuple, if None uses config reference

        Returns:
            ZTD value in meters at reference point and timestamp
        """
        # Load GNSS data
        gnss_path = os.path.join(self.config.download_dir, self.config.output_name)

        if not os.path.exists(gnss_path):
            log.error(f"GNSS data file not found: {gnss_path}")
            return None

        df = pd.read_csv(gnss_path)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Filter for time window around SAR timestamp
        time_window = timedelta(hours=self.config.temporal_buffer_hours)
        time_mask = (df['datetime'] >= sar_timestamp - time_window) & \
                   (df['datetime'] <= sar_timestamp + time_window)

        df_filtered = df[time_mask]

        if df_filtered.empty:
            log.warning(f"No GNSS data found within {self.config.temporal_buffer_hours} hours of {sar_timestamp}")
            return None

        # Spatial and temporal interpolation would go here
        # For now, return mean ZTD (simple baseline)
        mean_ztd = df_filtered['ztd'].mean()

        log.info(
            f"Computed reference ZTD for {sar_timestamp}: {mean_ztd:.4f} m "
            f"(from {len(df_filtered)} measurements)"
        )

        return mean_ztd
