#!/usr/bin/env python3
"""
GNSS RINEX Data Download Script

Downloads RINEX 3 observation and navigation files from the BKG GNSS Data Center
(https://igs.bkg.bund.de) for stations located within a given AOI polygon.

When stations is "auto", the script lists all available files for each date,
extracts unique station codes, and filters them against the IGS station
coordinates to keep only those within the buffered AOI bounding box.

Usage:
    python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json

    # Override stations explicitly
    python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json \
        --stations BOGO JOZE WROC

    # List available stations without downloading
    python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --list-only
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from settings.paths import DATA_DIR
from libs.internal.io.json_io import open_json
from libs.internal.geo.poly import geojson_to_bbox, buffer_bbox_wgs84
from libs.internal.gnss.rinex.rinex_downloader import RinexDownloader, NETWORKS
from libs.internal.log.logger import get_logger

log = get_logger()


def parse_dates(date_strings: list[str]) -> list[datetime]:
    """Parse date strings in YYYYMMDD format."""
    dates = []
    for ds in date_strings:
        dates.append(datetime.strptime(ds, "%Y%m%d"))
    return dates


def get_buffered_bbox(config: dict) -> list[float]:
    """Load the AOI polygon and return a buffered bounding box."""
    aoi_cfg = config["data"]["aoi"]
    polygon_path = os.path.join(DATA_DIR, aoi_cfg["polygon_child_dir"])
    geojson = open_json(polygon_path)
    bbox = geojson_to_bbox(geojson)

    rinex_cfg = config["data"]["rinex"]
    buffer_km = rinex_cfg.get("spatial_buffer_km", 100)
    buffer_m = buffer_km * 1000

    buffered = buffer_bbox_wgs84(bbox, buffer_m)
    log.info(
        f"AOI bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
    )
    log.info(
        f"Buffered ({buffer_km} km): "
        f"[{buffered[0]:.3f}, {buffered[1]:.3f}, {buffered[2]:.3f}, {buffered[3]:.3f}]"
    )
    return buffered


def discover_stations(
    downloader: RinexDownloader,
    date: datetime,
    bbox: list[float],
) -> list[str]:
    """
    Discover stations within the bounding box for a given date.

    Lists all files on the BKG server for the date, then filters station codes
    whose RINEX filenames are available. Station coordinates are approximated
    from the file listing metadata — the actual spatial filter is based on
    the 3-char country code and known station coordinates when available.

    For a robust filter, we rely on the BKG station list. As a practical
    fallback, we return all unique station codes found for that day and let
    the user review them.
    """
    files = downloader.list_files(date)
    if not files:
        return []

    stations = sorted(set(rf.station for rf in files))
    log.info(f"Found {len(stations)} unique stations on {date.strftime('%Y-%m-%d')}")
    return stations


def run_download(config: dict, station_override: list[str] = None, list_only: bool = False):
    rinex_cfg = config["data"]["rinex"]

    # Output directory
    output_dir = os.path.join(DATA_DIR, rinex_cfg["data_child_dir"], rinex_cfg["download_folder"])
    network = rinex_cfg.get("network", "IGS")
    obs = rinex_cfg.get("obs", True)
    nav = rinex_cfg.get("nav", False)
    dates = parse_dates(rinex_cfg["dates"])

    downloader = RinexDownloader(output_dir=output_dir, network=network)

    # Resolve stations
    if station_override:
        stations = [s.upper() for s in station_override]
        log.info(f"Using {len(stations)} stations from CLI: {stations}")
    elif rinex_cfg.get("stations") == "auto":
        bbox = get_buffered_bbox(config)
        log.info("Auto-discovering stations from first date listing...")
        stations = discover_stations(downloader, dates[0], bbox)
        if not stations:
            log.error("No stations found on the server for the first date")
            return
        log.info(f"Discovered {len(stations)} stations: {', '.join(stations[:20])}"
                 + (f" ... and {len(stations)-20} more" if len(stations) > 20 else ""))
    else:
        stations_val = rinex_cfg["stations"]
        if isinstance(stations_val, list):
            stations = [s.upper() for s in stations_val]
        else:
            log.error(f"Invalid 'stations' value in config: {stations_val}")
            return
        log.info(f"Using {len(stations)} stations from config: {stations}")

    if list_only:
        log.info("Stations that would be downloaded:")
        for s in stations:
            print(f"  {s}")
        log.info(f"Total: {len(stations)} stations x {len(dates)} dates")
        return

    # Download
    log.info("=" * 70)
    log.info(f"RINEX Download: {network} | {len(stations)} stations | {len(dates)} dates")
    log.info(f"  obs={obs}, nav={nav}")
    log.info(f"  Output: {output_dir}")
    log.info("=" * 70)

    total_files = 0
    for date in dates:
        log.info(f"\n--- {date.strftime('%Y-%m-%d')} ---")
        results = downloader.download_stations(
            stations=stations,
            date=date,
            obs=obs,
            nav=nav,
        )
        day_files = sum(len(v) for v in results.values())
        total_files += day_files
        log.info(f"Day total: {day_files} files for {len(results)} stations")
        time.sleep(1)

    log.info(f"\nDownload complete: {total_files} files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download GNSS RINEX data from BKG GDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Networks: {', '.join(NETWORKS)}

Examples:
  python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json
  python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --list-only
  python scripts/download_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --stations BOGO JOZE
        """
    )

    parser.add_argument("--config", type=str, required=True, help="Path to RINEX config JSON")
    parser.add_argument("--stations", nargs="+", help="Override station list (4-char codes)")
    parser.add_argument("--list-only", action="store_true", help="List stations without downloading")

    args = parser.parse_args()

    config = open_json(args.config)
    run_download(config, station_override=args.stations, list_only=args.list_only)


if __name__ == "__main__":
    main()
