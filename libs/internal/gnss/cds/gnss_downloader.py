try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from libs.internal.sbas.local_setup import local_setup
    local_setup()

import os
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from typing import Tuple, List
from pathlib import Path
from libs.internal.log.logger import get_logger
from libs.internal.gnss.cds.gnss_config import GnssConfig
from libs.internal.io.json_io import open_json
from settings.paths import KEYS_DIR

log = get_logger()


class GnssDownloader:
    """
    GNSS data downloader for Copernicus Climate Data Store.

    Downloads zenith tropospheric delay (ZTD) data from the CDS GNSS dataset
    for validation of atmospheric corrections in InSAR processing.

    Data source: https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss
    Provides ZTD and TCWV from IGS and EPN networks (1996-present).
    """

    def __init__(self, config_path: str):
        self.config = GnssConfig(config_path)
        self.cds_client = None

    def download_gnss_stations(
        self,
        aoi: gpd.GeoDataFrame,
        date_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """
        Download GNSS data from Copernicus Climate Data Store.

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
            f"Querying Copernicus CDS for GNSS data:\n"
            f"  Date range: {start_buffered} to {end_buffered}\n"
            f"  Spatial bounds: {bounds}"
        )

        # Query CDS API
        stations_data = self._query_cds_api(
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

    def _initialize_cds_client(self):
        """Initialize CDS API client with credentials from secrets file."""
        if self.cds_client is not None:
            return

        try:
            import cdsapi

            # Try to load credentials from secrets file
            try:
                secrets = open_json(os.path.join(KEYS_DIR, 'keys.json'))
                cds_config = secrets.get('cdsapi', {})

                # Support both 'key' and 'token' field names
                api_key = cds_config.get('key') or cds_config.get('token')
                api_url = cds_config.get('url')

                if api_url and api_key:
                    # Check if credentials are configured (not placeholder)
                    if api_key == "YOUR_UID:YOUR_API_KEY":
                        log.error("CDS API credentials not configured in .secrets/keys.json")
                        log.error("Please update the 'cdsapi' section with your credentials:")
                        log.error('  "url": "https://cds.climate.copernicus.eu/api"')
                        log.error('  "token": "YOUR_UID:YOUR_API_KEY"  (or use "key")')
                        log.error("\nGet credentials from: https://cds.climate.copernicus.eu/user")
                        raise ValueError("CDS API credentials not configured")

                    # Initialize client with credentials from secrets file
                    self.cds_client = cdsapi.Client(
                        url=api_url,
                        key=api_key,
                        verify=True
                    )
                    log.info("CDS API client initialized with credentials from .secrets/keys.json")
                else:
                    # Fall back to default ~/.cdsapirc
                    log.warning("CDS credentials not found in .secrets/keys.json")
                    log.warning("Attempting to use ~/.cdsapirc")
                    self.cds_client = cdsapi.Client()
                    log.info("CDS API client initialized from ~/.cdsapirc")

            except FileNotFoundError:
                # Fall back to default ~/.cdsapirc
                log.warning(".secrets/keys.json not found")
                log.warning("Attempting to use ~/.cdsapirc")
                self.cds_client = cdsapi.Client()
                log.info("CDS API client initialized from ~/.cdsapirc")

        except ImportError:
            log.error("cdsapi package not installed. Install with: pip install cdsapi")
            log.error("\nSetup options:")
            log.error("  1. Add credentials to .secrets/keys.json:")
            log.error('     "cdsapi": {')
            log.error('       "url": "https://cds.climate.copernicus.eu/api",')
            log.error('       "token": "YOUR_UID:YOUR_API_KEY"')
            log.error('     }')
            log.error("  2. Or configure ~/.cdsapirc")
            log.error("\nGet credentials from: https://cds.climate.copernicus.eu/user")
            raise
        except Exception as e:
            log.error(f"Failed to initialize CDS client: {e}")
            log.error("\nTroubleshooting:")
            log.error("  1. Register at: https://cds.climate.copernicus.eu/user/register")
            log.error("  2. Get your UID and API key from your user profile")
            log.error("  3. Add to .secrets/keys.json or ~/.cdsapirc")
            raise

    def _download_chunk(
        self,
        network_type: str,
        year: str,
        month: str,
        temp_file: str,
        bbox: List[float] = None,
        variables: List[str] = None,
        specific_days: List[str] = None
    ) -> bool:
        """
        Download GNSS data for a month or specific days.

        Args:
            network_type: Network type (e.g., 'epn_repro2', 'igs_repro3', 'igs_daily')
            year: Year as string
            month: Month as string (zero-padded)
            temp_file: Path to save the downloaded data
            bbox: Bounding box [north, west, south, east] (optional)
            variables: List of variables to download (optional)
            specific_days: List of specific days to download (optional, downloads full month if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine which days to download
            if specific_days:
                days = specific_days
            else:
                # Get all days in this month
                from calendar import monthrange
                _, days_in_month = monthrange(int(year), int(month))
                days = [str(d).zfill(2) for d in range(1, days_in_month + 1)]

            # Use provided variables or default to zenith_total_delay
            if not variables:
                variables = ['zenith_total_delay']

            request = {
                'network_type': network_type,
                'version': '1_0_0',
                'variable': variables,
                'year': [year],
                'month': [month],
                'day': days,
                'format': 'csv'
            }

            # Add bounding box if provided (CDS format: [north, west, south, east])
            if bbox:
                request['area'] = bbox

            if specific_days:
                log.info(f"  Downloading {year}-{month} ({len(days)} specific days: {', '.join(days)})...")
            else:
                log.info(f"  Downloading {year}-{month} ({len(days)} days)...")

            self.cds_client.retrieve(
                'insitu-observations-gnss',
                request,
                temp_file
            )
            return True

        except Exception as e:
            log.warning(f"  Failed to download {year}-{month}: {e}")
            return False

    def _query_cds_api(
        self,
        bounds: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Query Copernicus Climate Data Store for GNSS ZTD data.

        Downloads data in monthly chunks to avoid API size limits.
        Uses the CDS API to download GNSS troposphere data from IGS/EPN networks.
        Documentation: https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss

        Args:
            bounds: [minx, miny, maxx, maxy] in WGS84
            start_date: Start datetime
            end_date: End datetime

        Returns:
            DataFrame with GNSS ZTD data
        """
        try:
            self._initialize_cds_client()

            # Determine network type based on region
            # EPN for Europe, IGS for global coverage
            minlon, minlat, maxlon, maxlat = bounds
            is_europe = (-15 <= minlon <= 40) and (35 <= minlat <= 75)
            network_type = self.config.cds_network_type or ('epn_repro2' if is_europe else 'igs_repro3')

            # Convert bounds to CDS API format: [north, west, south, east]
            # Input bounds are: [minx, miny, maxx, maxy] = [west, south, east, north]
            cds_bbox = [maxlat, minlon, minlat, maxlon]

            # Get variables to download from config
            variables = self.config.cds_variables

            log.info(f"Using network: {network_type} (Europe: {is_europe})")
            log.info(f"Variables: {', '.join(variables)}")
            log.info(f"Spatial filter: N={maxlat:.2f}, W={minlon:.2f}, S={minlat:.2f}, E={maxlon:.2f}")

            # Determine which dates to download
            if self.config.dates:
                # Use specific dates from config (format: YYYYMMDD)
                specific_dates = [pd.to_datetime(d, format='%Y%m%d') for d in self.config.dates]
                log.info(f"Using {len(specific_dates)} specific dates from config:")
                for date in specific_dates[:5]:
                    log.info(f"  - {date.strftime('%Y%m%d')}")
                if len(specific_dates) > 5:
                    log.info(f"  ... and {len(specific_dates) - 5} more")

                # Group dates by year-month for efficient downloading
                date_groups = {}
                for date in specific_dates:
                    key = (str(date.year), str(date.month).zfill(2))
                    if key not in date_groups:
                        date_groups[key] = []
                    date_groups[key].append(str(date.day).zfill(2))

                months_to_download = list(date_groups.keys())
                specific_days = date_groups  # Dict of (year, month) -> [days]

            elif self.config.year:
                # Download entire year specified in config
                target_year = str(self.config.year)
                months_to_download = [(target_year, str(m).zfill(2)) for m in range(1, 13)]
                specific_days = None
                log.info(f"Using year from config: {target_year}")
                log.info(f"Downloading all 12 months of {target_year}")
            else:
                # Split date range into monthly chunks
                date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
                if len(date_range) == 0 or date_range[0] > start_date:
                    date_range = pd.DatetimeIndex([start_date]).append(date_range)

                months_to_download = []
                for date in date_range:
                    year = str(date.year)
                    month = str(date.month).zfill(2)
                    months_to_download.append((year, month))

                # Add the end month if not already included
                end_year = str(end_date.year)
                end_month = str(end_date.month).zfill(2)
                if (end_year, end_month) not in months_to_download:
                    months_to_download.append((end_year, end_month))

                specific_days = None
                log.info(f"Splitting request into {len(months_to_download)} monthly chunks:")
                log.info(f"  From: {months_to_download[0][0]}-{months_to_download[0][1]}")
                log.info(f"  To:   {months_to_download[-1][0]}-{months_to_download[-1][1]}")

            # Download each month separately
            all_data = []
            successful_downloads = 0

            for year, month in months_to_download:
                temp_file = os.path.join(self.config.download_dir, f'cds_temp_{year}_{month}.csv')

                # Get specific days for this month if available
                days_for_month = specific_days.get((year, month)) if specific_days else None

                if self._download_chunk(network_type, year, month, temp_file, bbox=cds_bbox,
                                       variables=variables, specific_days=days_for_month):
                    try:
                        # Try reading with different encodings
                        df_chunk = None
                        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                            try:
                                df_chunk = pd.read_csv(temp_file, encoding=encoding)
                                if encoding != 'utf-8':
                                    log.debug(f"  Loaded {year}-{month} using {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                continue

                        if df_chunk is None:
                            log.warning(f"  Failed to load {year}-{month}: Could not decode with any encoding")
                        else:
                            all_data.append(df_chunk)
                            successful_downloads += 1

                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        log.warning(f"  Failed to load {year}-{month}: {e}")

            if not all_data:
                log.error("No data downloaded successfully")
                return pd.DataFrame()

            log.info(f"Successfully downloaded {successful_downloads}/{len(months_to_download)} months")

            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)

            # Standardize column names
            # CDS format: station_name, latitude, longitude, observed_variable, observation_value, report_timestamp
            column_mapping = {
                'station_name': 'station_id',
                'latitude': 'lat',
                'longitude': 'lon',
                'report_timestamp': 'datetime'
            }

            df = df.rename(columns=column_mapping)

            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])

            # If multiple variables were requested, pivot to wide format
            if 'observed_variable' in df.columns and len(variables) > 1:
                log.info("Pivoting multi-variable data to wide format...")

                # Create a mapping for variable names to shorter column names
                var_name_map = {
                    'zenith_total_delay': 'ztd',
                    'total_column_water_vapour': 'tcwv',
                    'total_column_water_vapour_era5': 'tcwv_era5'
                }

                # Pivot the data
                df_pivot = df.pivot_table(
                    index=['station_id', 'lat', 'lon', 'datetime'],
                    columns='observed_variable',
                    values='observation_value',
                    aggfunc='first'
                ).reset_index()

                # Rename variable columns to shorter names
                df_pivot.columns = [var_name_map.get(col, col) if col in var_name_map else col
                                   for col in df_pivot.columns]

                df = df_pivot

            elif 'observation_value' in df.columns:
                # Single variable - just rename observation_value to ztd (or appropriate name)
                if 'zenith_total_delay' in variables or len(variables) == 1:
                    df['ztd'] = df['observation_value']
                    df = df.drop(columns=['observation_value'], errors='ignore')
                    if 'observed_variable' in df.columns:
                        df = df.drop(columns=['observed_variable'], errors='ignore')

            # Additional spatial filtering (CDS API should have filtered, but double-check)
            initial_count = len(df)
            spatial_mask = (
                (df['lat'] >= minlat) & (df['lat'] <= maxlat) &
                (df['lon'] >= minlon) & (df['lon'] <= maxlon)
            )
            df_filtered = df[spatial_mask].copy()
            if len(df_filtered) < initial_count:
                log.debug(f"Filtered out {initial_count - len(df_filtered)} records outside AOI bounds")

            # Determine which columns to keep (base + any data columns)
            base_cols = ['station_id', 'lat', 'lon', 'datetime']
            data_cols = [col for col in df_filtered.columns if col not in base_cols]
            required_cols = base_cols + data_cols

            # Remove records with missing data in key columns
            if 'ztd' in df_filtered.columns:
                df_filtered = df_filtered.dropna(subset=['ztd'])

            # Keep only required columns
            df_filtered = df_filtered[required_cols]

            log.info(f"Retrieved {len(df_filtered)} GNSS measurements")
            log.info(f"  Stations: {df_filtered['station_id'].nunique()}")
            log.info(f"  Date range: {df_filtered['datetime'].min()} to {df_filtered['datetime'].max()}")
            log.info(f"  Variables: {', '.join(data_cols)}")

            # Log range for each variable
            for col in data_cols:
                if col in df_filtered.columns and df_filtered[col].notna().any():
                    min_val = df_filtered[col].min()
                    max_val = df_filtered[col].max()
                    log.info(f"    {col}: {min_val:.4f} - {max_val:.4f}")

            return df_filtered

        except ImportError as e:
            log.error("CDS API not available. Install with: pip install cdsapi")
            log.info("\nAlternative: Use --method file to load from a CSV file")
            log.info("Or use --method synthetic for testing")
            return pd.DataFrame()

        except Exception as e:
            log.error(f"CDS API request failed: {e}")
            log.error("Make sure you have configured CDS API credentials:")
            log.error("  1. Register at https://cds.climate.copernicus.eu/user/register")
            log.error("  2. Get your UID and API key from your user profile")
            log.error("  3. Add to .secrets/keys.json:")
            log.error('     "cdsapi": {')
            log.error('       "url": "https://cds.climate.copernicus.eu/api",')
            log.error('       "token": "YOUR_UID:YOUR_API_KEY"')
            log.error('     }')
            log.error("     Or create ~/.cdsapirc with: url and key fields")
            log.info("\nAlternative: Use --method file or --method synthetic")
            import traceback
            traceback.print_exc()
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
