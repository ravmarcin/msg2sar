#!/usr/bin/env python3
"""
GNSS Data Download Script

Downloads GNSS tropospheric delay data for atmospheric correction validation.

Supports multiple methods:
1. EPOS GLASS Framework API (automatic, may require updates)
2. Manual file import (CSV, Nevada format, IGS format)
3. Synthetic data generation (for testing)

Usage:
    # Try API download (may not work if API has changed)
    python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method api

    # Load from CSV file
    python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method file --file path/to/gnss_data.csv

    # Generate synthetic data (for testing)
    python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method synthetic

Manual download sources:
    - Nevada Geodetic Laboratory: http://geodesy.unr.edu/
    - EUREF: http://www.epncb.oma.be/
    - IGS: https://igs.org/
    - GNSS Data Portal: https://gnssdata-epos.oca.eu/
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
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import numpy as np
import pandas as pd
from utils.internal.gnss import GnssDownloader, GnssConfig
from utils.internal.log.logger import get_logger

log = get_logger()


def download_via_api(config_path: str):
    """
    Download GNSS data via EPOS API.

    Args:
        config_path: Path to config file
    """
    log.info("="*80)
    log.info("GNSS Data Download via EPOS API")
    log.info("="*80)

    downloader = GnssDownloader(config_path)

    # Get date range from config or use defaults
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    log.info(f"Attempting to download GNSS data from EPOS...")
    log.info(f"Date range: {start_date} to {end_date}")

    try:
        gnss_data = downloader.download_gnss_stations(
            aoi=downloader.config.aoi,
            date_range=(start_date, end_date)
        )

        if not gnss_data.empty:
            log.info(f"\n✓ Successfully downloaded {len(gnss_data)} measurements")
            log.info(f"  Stations: {gnss_data['station_id'].nunique()}")
            log.info(f"  Date range: {gnss_data['datetime'].min()} to {gnss_data['datetime'].max()}")
        else:
            log.error("\n✗ No data retrieved from API")
            log.info("\nAlternatives:")
            log.info("  1. Use --method file to load from a file")
            log.info("  2. Use --method synthetic for testing")
            log.info("  3. Manually download from:")
            log.info("     - Nevada: http://geodesy.unr.edu/")
            log.info("     - EUREF: http://www.epncb.oma.be/")
            log.info("     - IGS: https://igs.org/")

    except Exception as e:
        log.error(f"Download failed: {e}")
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


def generate_synthetic(config_path: str):
    """
    Generate synthetic GNSS data for testing.

    Args:
        config_path: Path to config file
    """
    log.info("="*80)
    log.info("Generate Synthetic GNSS Data")
    log.info("="*80)

    config = GnssConfig(config_path)

    # Get AOI bounds
    bounds = config.aoi.total_bounds  # minx, miny, maxx, maxy

    log.info(f"Generating synthetic data for AOI: {bounds}")

    # Generate 5-10 synthetic stations
    n_stations = np.random.randint(5, 11)

    # Create stations within AOI
    stations = []
    for i in range(n_stations):
        lat = np.random.uniform(bounds[1], bounds[3])
        lon = np.random.uniform(bounds[0], bounds[2])
        stations.append({
            'id': f'SYN{i+1:02d}',
            'lat': lat,
            'lon': lon
        })

    # Generate ZTD time series for each station
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Daily samples
    dates = pd.date_range(start_date, end_date, freq='D')

    records = []
    for station in stations:
        # Base ZTD around 2.3 meters with daily variation
        base_ztd = 2.3
        daily_variation = 0.1

        for date in dates:
            # Add seasonal variation (more water vapor in summer)
            day_of_year = date.timetuple().tm_yday
            seasonal = 0.15 * np.sin(2 * np.pi * day_of_year / 365)

            # Add random noise
            noise = np.random.normal(0, 0.02)

            ztd = base_ztd + seasonal + daily_variation * np.random.randn() + noise

            records.append({
                'station_id': station['id'],
                'lat': station['lat'],
                'lon': station['lon'],
                'ztd': ztd,
                'datetime': date
            })

    df = pd.DataFrame(records)

    # Save to output location
    import os
    output_path = os.path.join(config.download_dir, config.output_name)
    df.to_csv(output_path, index=False)

    log.info(f"\n✓ Generated {len(df)} synthetic measurements")
    log.info(f"  Stations: {n_stations}")
    log.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    log.info(f"  ZTD range: {df['ztd'].min():.4f} - {df['ztd'].max():.4f} m")
    log.info(f"  Saved to: {output_path}")

    log.info(f"\nSample data:")
    print(df.head())

    log.warning("\nNote: This is SYNTHETIC data for testing only!")
    log.warning("Do not use for scientific validation.")


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

    print("\n4. EPOS GNSS Data Portal")
    print("   URL: https://gnssdata-epos.oca.eu/")
    print("   Steps:")
    print("   - Use web interface to browse stations")
    print("   - Download processed troposphere products")
    print("   - Convert to CSV format")

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
        description="Download GNSS tropospheric delay data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Try API download
  python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method api

  # Load from CSV
  python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method file --file gnss_data.csv

  # Generate test data
  python scripts/download_gnss_data.py --config data/configs/gnss/2023/bogo_pl.json --method synthetic

  # Show manual download instructions
  python scripts/download_gnss_data.py --help-manual
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
        choices=['api', 'file', 'synthetic'],
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

    elif args.method == 'synthetic':
        generate_synthetic(args.config)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
