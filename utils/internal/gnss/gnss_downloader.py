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
        Query EPOS GNSS Data Portal (GLASS Framework) API.

        The GLASS Framework provides access to GNSS station data including
        tropospheric delays. API documentation:
        https://gnssdata-epos.oca.eu/GlassFramework/

        Args:
            bounds: [minx, miny, maxx, maxy] in WGS84
            start_date: Start datetime
            end_date: End datetime

        Returns:
            DataFrame with GNSS ZTD data
        """
        # EPOS GLASS Framework API endpoints
        base_url = self.config.epos_api_url.rstrip('/')

        # First, get list of stations in the area
        stations_url = f"{base_url}/api/gnss/stations"

        # Build query for stations
        station_params = {
            'minLat': bounds[1],
            'maxLat': bounds[3],
            'minLon': bounds[0],
            'maxLon': bounds[2]
        }

        try:
            log.info(f"Querying EPOS GLASS Framework for stations in bounds: {bounds}")
            log.info(f"API URL: {stations_url}")

            # Get stations
            response = requests.get(stations_url, params=station_params, timeout=60)
            response.raise_for_status()
            stations_data = response.json()

            # Parse station list
            stations = []
            if isinstance(stations_data, list):
                stations = stations_data
            elif isinstance(stations_data, dict) and 'stations' in stations_data:
                stations = stations_data['stations']
            else:
                log.warning(f"Unexpected stations response format: {type(stations_data)}")
                stations = []

            if not stations:
                log.warning("No GNSS stations found in the specified area")
                return pd.DataFrame()

            log.info(f"Found {len(stations)} GNSS stations")

            # For each station, get ZTD data
            all_records = []

            for station in stations[:20]:  # Limit to 20 stations to avoid too many requests
                station_id = station.get('id') or station.get('stationId') or station.get('name')
                if not station_id:
                    continue

                # Get ZTD data for this station
                ztd_url = f"{base_url}/api/gnss/troposphere"
                ztd_params = {
                    'station': station_id,
                    'startTime': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'endTime': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'format': 'json'
                }

                try:
                    log.debug(f"Querying ZTD for station {station_id}")
                    ztd_response = requests.get(ztd_url, params=ztd_params, timeout=30)

                    if ztd_response.status_code == 200:
                        ztd_data = ztd_response.json()

                        # Parse ZTD data
                        station_lat = station.get('latitude') or station.get('lat')
                        station_lon = station.get('longitude') or station.get('lon')

                        if isinstance(ztd_data, list):
                            for record in ztd_data:
                                all_records.append({
                                    'station_id': station_id,
                                    'lat': station_lat,
                                    'lon': station_lon,
                                    'ztd': record.get('ztd', record.get('value')),
                                    'datetime': pd.to_datetime(record.get('datetime', record.get('timestamp')))
                                })
                        elif isinstance(ztd_data, dict):
                            # Single record or different format
                            all_records.append({
                                'station_id': station_id,
                                'lat': station_lat,
                                'lon': station_lon,
                                'ztd': ztd_data.get('ztd', ztd_data.get('value')),
                                'datetime': pd.to_datetime(ztd_data.get('datetime', ztd_data.get('timestamp')))
                            })

                except requests.exceptions.RequestException as e:
                    log.warning(f"Failed to get ZTD for station {station_id}: {e}")
                    continue

            if not all_records:
                log.warning("No ZTD data retrieved from any station")
                return pd.DataFrame()

            df = pd.DataFrame(all_records)

            # Remove records with missing ZTD
            df = df.dropna(subset=['ztd'])

            log.info(f"Retrieved {len(df)} GNSS ZTD measurements from {df['station_id'].nunique()} stations")

            return df

        except requests.exceptions.RequestException as e:
            log.error(f"EPOS API request failed: {e}")
            log.error(f"URL attempted: {stations_url}")
            log.error(f"This may be because the EPOS GLASS API structure has changed.")
            log.error(f"Please check: {base_url}")
            log.info(f"\nAlternative: You can manually download GNSS data from:")
            log.info(f"  - Nevada Geodetic Laboratory: http://geodesy.unr.edu/")
            log.info(f"  - EUREF: http://www.epncb.oma.be/")
            log.info(f"  - IGS: https://igs.org/")
            return pd.DataFrame()

        except (KeyError, ValueError) as e:
            log.error(f"Failed to parse EPOS API response: {e}")
            return pd.DataFrame()

    def load_gnss_from_file(
        self,
        filepath: str,
        file_format: str = 'auto'
    ) -> pd.DataFrame:
        """
        Load GNSS ZTD data from a file (alternative to API download).

        Supports multiple formats:
        - CSV: pre-formatted GNSS data
        - Nevada Geodetic Lab format
        - IGS ZTD format
        - Custom format with configurable columns

        Args:
            filepath: Path to GNSS data file
            file_format: Format type ('csv', 'nevada', 'igs', 'auto')

        Returns:
            DataFrame with GNSS ZTD data
        """
        log.info(f"Loading GNSS data from file: {filepath}")

        if not os.path.exists(filepath):
            log.error(f"File not found: {filepath}")
            return pd.DataFrame()

        try:
            if file_format == 'auto':
                # Auto-detect format based on file extension
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.csv':
                    file_format = 'csv'
                elif ext == '.ztd':
                    file_format = 'igs'
                else:
                    file_format = 'csv'

            if file_format == 'csv':
                # Standard CSV format
                df = pd.read_csv(filepath)

                # Ensure required columns exist
                required_cols = ['station_id', 'lat', 'lon', 'ztd', 'datetime']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    log.error(f"CSV missing required columns: {missing_cols}")
                    log.info(f"Expected columns: {required_cols}")
                    log.info(f"Found columns: {list(df.columns)}")
                    return pd.DataFrame()

                # Convert datetime column
                df['datetime'] = pd.to_datetime(df['datetime'])

            elif file_format == 'nevada':
                # Nevada Geodetic Laboratory format
                # Format: DATE, LAT, LON, ZTD_VALUE
                df = pd.read_csv(
                    filepath,
                    delim_whitespace=True,
                    names=['datetime', 'lat', 'lon', 'ztd'],
                    comment='#'
                )
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['station_id'] = os.path.splitext(os.path.basename(filepath))[0]

            elif file_format == 'igs':
                # IGS ZTD format (simplified parser)
                records = []
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 4:
                            records.append({
                                'datetime': pd.to_datetime(parts[0]),
                                'ztd': float(parts[1]),
                                'lat': float(parts[2]) if len(parts) > 2 else None,
                                'lon': float(parts[3]) if len(parts) > 3 else None
                            })
                df = pd.DataFrame(records)
                df['station_id'] = os.path.splitext(os.path.basename(filepath))[0]

            else:
                log.error(f"Unsupported file format: {file_format}")
                return pd.DataFrame()

            log.info(f"Loaded {len(df)} GNSS measurements from file")
            log.info(f"  Stations: {df['station_id'].nunique()}")
            log.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

            # Save to standard location
            output_path = os.path.join(self.config.download_dir, self.config.output_name)
            df.to_csv(output_path, index=False)
            log.info(f"Saved to: {output_path}")

            return df

        except Exception as e:
            log.error(f"Failed to load GNSS data from file: {e}")
            import traceback
            traceback.print_exc()
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
