#!/usr/bin/env python3
"""
GNSS Water Vapour and ZTD Data Download Script

Downloads GNSS-derived water vapour (TCWV) and zenith tropospheric delay (ZTD) data
from ECMWF's Copernicus Climate Data Store for atmospheric correction validation.

Data source: IGS and EPN GNSS networks (1996-present)
Provider: ECMWF via Copernicus Climate Data Store (CDS)
Variables: Zenith Total Delay, Total Column Water Vapour, TCWV ERA5

Supports multiple methods:
1. Copernicus CDS API (automatic, requires CDS API key)
2. Manual file import (CSV, Nevada format, IGS format)

Configuration:
    Dates can be specified in config using one of:
    - "dates": ["20220115", "20220220", ...]  (specific dates)
    - "year": 2022  (entire year)
    - If neither specified, uses default date range

Usage:
    # API download (requires CDS credentials in .secrets/keys.json)
    python scripts/download_gnss_data.py --config data/configs/gnss/2022/bogo_pl.json --method api

    # Load from CSV file
    python scripts/download_gnss_data.py --config data/configs/gnss/2022/bogo_pl.json --method file --file path/to/gnss_data.csv

CDS API Setup:
    1. Register at https://cds.climate.copernicus.eu/user/register
    2. Get your UID and API key from your profile
    3. Add to .secrets/keys.json:
       {
         "cdsapi": {
           "url": "https://cds.climate.copernicus.eu/api",
           "token": "UID:API_KEY"
         }
       }

Data Format:
    Output CSV contains: station_id, lat, lon, datetime, ztd, tcwv, tcwv_era5
    - ztd: Zenith Total Delay (meters)
    - tcwv: Total Column Water Vapour from GNSS (kg/m²)
    - tcwv_era5: Total Column Water Vapour from ERA5 reanalysis (kg/m²)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError:
    from libs.internal.sbas.local_setup import local_setup
    local_setup()

import numpy as np
import pandas as pd
from libs.internal.gnss import GnssDownloader, GnssConfig
from libs.internal.log.logger import get_logger

log = get_logger()


def download_via_api(config_path: str):
    """
    Download GNSS water vapour and ZTD data via Copernicus Climate Data Store API.

    Date specification priority:
    1. config.dates - specific dates (e.g., ["20220115", "20220220"])
    2. config.year - entire year (e.g., 2022)
    3. Default fallback - uses dummy date range (ignored if dates/year in config)

    Args:
        config_path: Path to config file
    """
    log.info("="*80)
    log.info("GNSS Water Vapour & ZTD Download via ECMWF Copernicus CDS")
    log.info("="*80)

    downloader = GnssDownloader(config_path)

    # Date range is only used as fallback if config doesn't specify dates or year
    # The actual dates are determined by config.dates or config.year
    if downloader.config.dates:
        log.info(f"Using {len(downloader.config.dates)} specific dates from config")
    elif downloader.config.year:
        log.info(f"Using year {downloader.config.year} from config (all 12 months)")
    else:
        log.warning("No dates or year specified in config, using default 2022")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        log.info(f"Date range fallback: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")

    log.info(f"Downloading from ECMWF Copernicus CDS...")

    try:
        # Use dummy dates - actual dates come from config
        gnss_data = downloader.download_gnss_stations(
            aoi=downloader.config.aoi,
            date_range=(datetime(2022, 1, 1), datetime(2022, 1, 1))  # Ignored if config has dates/year
        )

        if not gnss_data.empty:
            log.info(f"\n✓ Successfully downloaded {len(gnss_data)} measurements")
            log.info(f"  Stations: {gnss_data['station_id'].nunique()}")
            log.info(f"  Date range: {gnss_data['datetime'].min()} to {gnss_data['datetime'].max()}")

            # Show which variables were downloaded
            data_cols = [col for col in gnss_data.columns if col not in ['station_id', 'lat', 'lon', 'datetime']]
            log.info(f"  Variables: {', '.join(data_cols)}")

            output_path = f"{downloader.config.download_dir}/{downloader.config.output_name}"
            log.info(f"  Saved to: {output_path}")
        else:
            log.error("\n✗ No data retrieved from CDS API")
            log.info("\nTroubleshooting:")
            log.info("  1. Check CDS credentials in .secrets/keys.json")
            log.info("  2. Verify dates/year in config are available")
            log.info("  3. Check network connection")
            log.info("\nAlternatives:")
            log.info("  - Use --method file to load pre-downloaded data")

    except Exception as e:
        log.error(f"Download failed: {e}")
        log.info("\nTroubleshooting:")
        log.info("  1. Install cdsapi: pip install cdsapi")
        log.info("  2. Configure CDS credentials in .secrets/keys.json:")
        log.info('     "cdsapi": {"url": "https://cds.climate.copernicus.eu/api", "token": "UID:KEY"}')
        log.info("  3. Register at: https://cds.climate.copernicus.eu/user/register")
        log.info("  4. Accept the license for 'GNSS' dataset on CDS website")
        import traceback
        traceback.print_exc()


def load_from_file(config_path: str, filepath: str, file_format: str = 'auto'):
    """
    Load GNSS data from file.

    Args:
        config_path: Path to config file
        filepath: Path to GNSS data file
        file_format: File format ('csv', 'nevada', 'igs', 'auto')
    """
    log.info("="*80)
    log.info("GNSS Data Load from File")
    log.info("="*80)

    downloader = GnssDownloader(config_path)

    log.info(f"Loading from: {filepath}")
    log.info(f"Format: {file_format}")

    gnss_data = downloader.load_gnss_from_file(filepath, file_format)

    if not gnss_data.empty:
        log.info(f"\n✓ Successfully loaded {len(gnss_data)} measurements")
        log.info(f"  Stations: {gnss_data['station_id'].nunique()}")
        log.info(f"  Date range: {gnss_data['datetime'].min()} to {gnss_data['datetime'].max()}")

        # Show sample
        log.info(f"\nSample data:")
        print(gnss_data.head())
    else:
        log.error("\n✗ Failed to load data from file")


def print_manual_instructions():
    """Print instructions for manual GNSS data download."""
    print("\n" + "="*80)
    print("Manual GNSS Data Download Instructions")
    print("="*80)

    print("\n1. Nevada Geodetic Laboratory (UNR)")
    print("   URL: http://geodesy.unr.edu/")
    print("   Steps:")
    print("   - Select 'Data' -> 'Data Archive'")
    print("   - Choose stations in your region")
    print("   - Download tropospheric delay files (.trop or .ztd)")
    print("   - Convert to CSV format with columns: station_id, lat, lon, ztd, datetime")

    print("\n2. EUREF Permanent GNSS Network")
    print("   URL: http://www.epncb.oma.be/")
    print("   Steps:")
    print("   - Select 'Products' -> 'Troposphere'")
    print("   - Choose date range and stations")
    print("   - Download ZTD products")
    print("   - Parse and convert to standard CSV format")

    print("\n3. International GNSS Service (IGS)")
    print("   URL: https://igs.org/")
    print("   Steps:")
    print("   - Navigate to 'Data & Products'")
    print("   - Download troposphere products")
    print("   - Convert to CSV format")

    print("\n4. ECMWF Copernicus Climate Data Store (Recommended)")
    print("   URL: https://cds.climate.copernicus.eu/datasets/insitu-observations-gnss")
    print("   Steps:")
    print("   - Register for free account and get API credentials")
    print("   - Add credentials to .secrets/keys.json")
    print("   - Use automated download with: --method api")
    print("   - Downloads ZTD, TCWV, and TCWV ERA5 automatically")
    print("   - Or download manually from website and use: --method file")

    print("\nStandard CSV format required:")
    print("  Columns: station_id, lat, lon, ztd, datetime")
    print("  Example:")
    print("    station_id,lat,lon,ztd,datetime")
    print("    WROC,51.113,17.062,2.345,2023-06-07T04:44:45")
    print("    BOGO,51.665,21.044,2.389,2023-06-07T04:44:45")

    print("\nOnce you have the CSV file, load it with:")
    print("  python scripts/download_gnss_data.py --config <config> --method file --file <path_to_csv>")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download GNSS water vapour and ZTD data from ECMWF Copernicus CDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Copernicus CDS (requires credentials in .secrets/keys.json)
  python scripts/download_gnss_data.py --config data/configs/gnss/2022/bogo_pl.json --method api

  # Load from pre-downloaded CSV
  python scripts/download_gnss_data.py --config data/configs/gnss/2022/bogo_pl.json --method file --file gnss_data.csv

  # Show manual download instructions
  python scripts/download_gnss_data.py --help-manual

Config should specify dates using:
  - "dates": ["20220115", "20220220", ...] for specific dates
  - "year": 2022 for entire year
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required='--help-manual' not in sys.argv,
        help='Path to GNSS config file'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['api', 'file'],
        default='api',
        help='Download method (default: api)'
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Path to GNSS data file (required if method=file)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['auto', 'csv', 'nevada', 'igs'],
        default='auto',
        help='File format (default: auto)'
    )

    parser.add_argument(
        '--help-manual',
        action='store_true',
        help='Show manual download instructions'
    )

    args = parser.parse_args()

    if args.help_manual:
        print_manual_instructions()
        return

    # Execute based on method
    if args.method == 'api':
        download_via_api(args.config)

    elif args.method == 'file':
        if not args.file:
            log.error("--file argument required when using --method file")
            return
        load_from_file(args.config, args.file, args.format)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
