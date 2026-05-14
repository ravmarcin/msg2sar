#!/usr/bin/env python3
"""
GNSS RINEX Data Processing Script

Processes downloaded RINEX 3 observation and navigation files using the gnsspy library.
Supports: file listing, observation export, SPP positioning with precise orbits,
multipath analysis, and observation-derived tropospheric delay estimation.

Usage:
    # List available RINEX files
    python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --list

    # Export raw observations at 30s resolution
    python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --summary

    # Run SPP with precise orbits (downloads SP3/CLK, computes per-epoch positions + tropo)
    python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --spp

    # Compute multipath indicators (MP1, MP2)
    python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --multipath

    # Run all processing steps
    python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --all
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from settings.paths import DATA_DIR
from libs.internal.io.json_io import open_json
from libs.internal.gnss.rinex.rinex_processor import GnssRinexProcessor
from libs.internal.gnss.rinex.orbit_downloader import download_orbits
from libs.internal.log.logger import get_logger

log = get_logger()


def get_rinex_dir(config: dict) -> str:
    """Resolve the RINEX data directory from config."""
    rinex_cfg = config["data"]["rinex"]
    return os.path.join(DATA_DIR, rinex_cfg["data_child_dir"], rinex_cfg["download_folder"])


def get_output_dir(config: dict) -> str:
    """Resolve the output directory for processed results."""
    rinex_cfg = config["data"]["rinex"]
    output_dir = os.path.join(DATA_DIR, rinex_cfg["data_child_dir"], "processed")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_orbit_dir(config: dict) -> str:
    """Resolve the orbit products directory."""
    rinex_cfg = config["data"]["rinex"]
    orbit_dir = os.path.join(DATA_DIR, rinex_cfg["data_child_dir"], "orbits")
    os.makedirs(orbit_dir, exist_ok=True)
    return orbit_dir


def list_files(processor: GnssRinexProcessor):
    """List RINEX files grouped by type, deduplicated to .gz originals only."""
    files = processor.list_rinex_files()

    # Deduplicate: prefer .crx.gz for obs, .rnx.gz for nav (skip derived .crx/.rnx)
    files["observation"] = [
        f for f in files["observation"]
        if f.endswith(".gz")
    ]
    files["navigation"] = [
        f for f in files["navigation"]
        if f.endswith(".gz")
    ]

    log.info("Observation files:")
    for f in files["observation"]:
        log.info(f"  {f}")

    log.info("Navigation files:")
    for f in files["navigation"]:
        log.info(f"  {f}")

    if files["other"]:
        log.info("Other files:")
        for f in files["other"]:
            log.info(f"  {f}")

    return files


def run_summary(processor: GnssRinexProcessor, files: dict, output_dir: str, spp_results: dict = None):
    """
    Export observations at 30s resolution and daily station summary.

    When SPP results are available, uses SPP-derived mean position and tropospheric
    delay instead of the static RINEX header values.
    """
    import numpy as np

    daily_rows = []

    for obs_file in files["observation"]:
        log.info(f"\n--- Processing: {obs_file} ---")

        try:
            obs = processor.read_observation(obs_file)
            station = obs_file[:4]
            date_str = obs.epoch.strftime("%Y%m%d") if hasattr(obs.epoch, "strftime") else str(obs.epoch)

            # Use SPP-derived position if available, otherwise fall back to header
            spp_key = f"{station}_{date_str}"
            if spp_results and spp_key in spp_results:
                spp_df = spp_results[spp_key]
                position = {
                    "lat": float(spp_df["lat"].mean()),
                    "lon": float(spp_df["lon"].mean()),
                    "height": float(spp_df["height_m"].mean()),
                    "X": float(spp_df["X"].mean()),
                    "Y": float(spp_df["Y"].mean()),
                    "Z": float(spp_df["Z"].mean()),
                }
                tropo_mean = float(spp_df["tropo_mean_m"].mean())
                tropo_std = float(spp_df["tropo_mean_m"].std())
                height_std = float(spp_df["height_m"].std())
                n_spp_epochs = len(spp_df)
                position_source = "spp"
            else:
                position = processor.get_station_position(obs_file)
                tropo_mean = np.nan
                tropo_std = np.nan
                height_std = np.nan
                n_spp_epochs = 0
                position_source = "header"

            # --- 30s epoch resolution: one CSV per observation file ---
            df = obs.observation.reset_index()
            df.insert(0, "station", station)
            df["lat"] = position["lat"]
            df["lon"] = position["lon"]
            df["height_m"] = position["height"]
            df.rename(columns={"Epoch": "datetime", "SV": "satellite"}, inplace=True)

            csv_name = f"{station}_{date_str}_obs.csv"
            csv_path = os.path.join(output_dir, csv_name)
            df.to_csv(csv_path, index=False)

            log.info(
                f"  Saved {len(df)} records ({df['datetime'].nunique()} epochs x "
                f"{df['satellite'].nunique()} satellites): {csv_path}"
            )

            # --- Daily summary row ---
            summary = processor.get_observation_summary(obs_file)
            row = {
                "station": station,
                "date": summary["epoch"],
                "lat": position["lat"],
                "lon": position["lon"],
                "height_m": position["height"],
                "height_std_m": height_std,
                "tropo_mean_m": tropo_mean,
                "tropo_std_m": tropo_std,
                "position_source": position_source,
                "n_spp_epochs": n_spp_epochs,
                "version": summary["version"],
                "receiver": str(summary["receiver"]),
                "antenna": str(summary["antenna"]),
                "interval_s": summary["interval_s"],
                "n_epochs": summary["n_epochs"],
                "time_start": summary["time_start"],
                "time_end": summary["time_end"],
                "n_satellites": summary["n_satellites"],
                "systems": ",".join(summary["systems"]),
            }
            daily_rows.append(row)

        except Exception as e:
            log.error(f"  Failed to process {obs_file}: {e}")

    # --- Save daily summary ---
    if daily_rows:
        df_daily = pd.DataFrame(daily_rows)
        csv_path = os.path.join(output_dir, "station_daily_summary.csv")
        df_daily.to_csv(csv_path, index=False)
        log.info(f"\nSaved daily summary ({len(df_daily)} days): {csv_path}")


def run_spp(processor: GnssRinexProcessor, files: dict, output_dir: str, orbit_dir: str) -> dict:
    """
    Run SPP with precise orbits: per-epoch positioning with observation-derived
    tropospheric correction.

    Downloads SP3 and clock files from GFZ, interpolates orbits, and computes
    per-epoch positions using ionosphere-free combination.

    Returns:
        Dict mapping "{station}_{date}" to SPP DataFrame for use in summary.
    """
    import numpy as np

    results = {}

    for obs_file in files["observation"]:
        log.info(f"\n--- SPP: {obs_file} ---")

        try:
            obs = processor.read_observation(obs_file)
            station = obs_file[:4]
            epoch_date = obs.epoch
            if hasattr(epoch_date, "date"):
                epoch_date = epoch_date.date()

            date_str = epoch_date.strftime("%Y%m%d")

            # Download orbit products for this date
            orbit_files = download_orbits(epoch_date, orbit_dir, product="gfz")
            missing = [k for k, v in orbit_files.items() if v is None]
            if missing:
                log.error(f"  Missing orbit files: {missing}. Skipping SPP.")
                continue

            # Run SPP (orbit interpolation + positioning)
            position = processor.compute_spp_with_orbits(
                obs_file, orbit_dir,
                system="G", cut_off=7.0,
                sp3_product="gfz", clock_product="gfz",
                interval=obs.interval,
            )

            if position is None or position.empty:
                log.error(f"  SPP returned no solutions for {obs_file}")
                continue

            # Convert ECEF to geodetic for each epoch
            a = 6378137.0
            f = 1 / 298.257223563
            e2 = 2 * f - f ** 2

            x_arr = position["X"].values.astype(float)
            y_arr = position["Y"].values.astype(float)
            z_arr = position["Z"].values.astype(float)

            lon_arr = np.degrees(np.arctan2(y_arr, x_arr))
            p_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
            lat_arr = np.degrees(np.arctan2(z_arr, p_arr * (1 - e2)))

            for _ in range(5):
                N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat_arr)) ** 2)
                lat_arr = np.degrees(np.arctan2(z_arr + e2 * N * np.sin(np.radians(lat_arr)), p_arr))

            N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat_arr)) ** 2)
            height_arr = p_arr / np.cos(np.radians(lat_arr)) - N

            position["lat"] = lat_arr
            position["lon"] = lon_arr
            position["height_m"] = height_arr
            position["station"] = station

            # Save per-epoch positions
            csv_name = f"{station}_{date_str}_spp.csv"
            csv_path = os.path.join(output_dir, csv_name)
            position.to_csv(csv_path)
            log.info(f"  Saved SPP ({len(position)} epochs): {csv_path}")

            results[f"{station}_{date_str}"] = position

        except Exception as e:
            log.error(f"  SPP failed for {obs_file}: {e}")
            import traceback
            traceback.print_exc()

    return results


def run_multipath(processor: GnssRinexProcessor, files: dict, output_dir: str):
    """
    Compute multipath indicators (MP1, MP2) for each observation file.

    Multipath is computed from dual-frequency observations per satellite per epoch.
    """
    for obs_file in files["observation"]:
        log.info(f"\n--- Multipath: {obs_file} ---")

        try:
            obs = processor.read_observation(obs_file)
            station = obs_file[:4]
            epoch_date = obs.epoch
            if hasattr(epoch_date, "date"):
                epoch_date = epoch_date.date()
            date_str = epoch_date.strftime("%Y%m%d")

            mp = processor.compute_multipath(obs_file, system="G")

            if mp is None or mp.empty:
                log.error(f"  Multipath returned no data for {obs_file}")
                continue

            mp_out = mp.reset_index()
            mp_out.insert(0, "station", station)

            csv_name = f"{station}_{date_str}_multipath.csv"
            csv_path = os.path.join(output_dir, csv_name)
            mp_out.to_csv(csv_path, index=False)
            log.info(f"  Saved multipath ({len(mp_out)} records): {csv_path}")

        except Exception as e:
            log.error(f"  Multipath failed for {obs_file}: {e}")
            import traceback
            traceback.print_exc()


def _load_existing_spp(files: dict, output_dir: str) -> dict:
    """Load existing SPP CSV results from a previous run."""
    results = {}
    for obs_file in files["observation"]:
        station = obs_file[:4]
        # Extract date from filename (position 12-19 in RINEX 3 long name)
        try:
            # Try to find matching SPP file by station prefix
            import glob
            pattern = os.path.join(output_dir, f"{station}_*_spp.csv")
            for spp_path in glob.glob(pattern):
                spp_df = pd.read_csv(spp_path, index_col="Epoch", parse_dates=True)
                # Extract key from filename: BOGO_20210410_spp.csv -> BOGO_20210410
                key = os.path.basename(spp_path).replace("_spp.csv", "")
                results[key] = spp_df
        except Exception:
            pass
    if results:
        log.info(f"Loaded {len(results)} existing SPP result(s) for daily summary")
    return results


def run_processing(config: dict, do_list: bool, do_summary: bool, do_spp: bool, do_multipath: bool):
    rinex_dir = get_rinex_dir(config)
    output_dir = get_output_dir(config)
    orbit_dir = get_orbit_dir(config)

    log.info("=" * 70)
    log.info("RINEX Data Processing")
    log.info(f"  Data directory: {rinex_dir}")
    log.info(f"  Output directory: {output_dir}")
    log.info(f"  Orbit directory: {orbit_dir}")
    log.info("=" * 70)

    processor = GnssRinexProcessor(rinex_dir)
    files = list_files(processor)

    if not files["observation"] and not files["navigation"]:
        log.error("No RINEX files found in the data directory")
        return

    if do_list and not do_summary and not do_spp and not do_multipath:
        return

    # Run SPP first so results can be used in the daily summary
    spp_results = {}
    if do_spp:
        spp_results = run_spp(processor, files, output_dir, orbit_dir)

    if do_summary:
        # If SPP wasn't run this session, try loading existing SPP CSVs
        if not spp_results:
            spp_results = _load_existing_spp(files, output_dir)
        run_summary(processor, files, output_dir, spp_results=spp_results)

    if do_multipath:
        run_multipath(processor, files, output_dir)

    log.info("\nProcessing complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Process GNSS RINEX data using gnsspy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --list
  python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --summary
  python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --spp
  python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --multipath
  python scripts/process_rinex_data.py --config data/configs/gnss/rinex/2021/bogo_pl.json --all
        """
    )

    parser.add_argument("--config", type=str, required=True, help="Path to RINEX config JSON")
    parser.add_argument("--list", action="store_true", help="List available RINEX files")
    parser.add_argument("--summary", action="store_true", help="Export observations at 30s resolution")
    parser.add_argument("--spp", action="store_true", help="Run SPP with precise orbits (downloads SP3/CLK)")
    parser.add_argument("--multipath", action="store_true", help="Compute multipath indicators (MP1, MP2)")
    parser.add_argument("--all", action="store_true", help="Run all processing steps")

    args = parser.parse_args()

    config = open_json(args.config)

    do_list = args.list or args.all
    do_summary = args.summary or args.all
    do_spp = args.spp or args.all
    do_multipath = args.multipath or args.all

    if not any([do_list, do_summary, do_spp, do_multipath]):
        parser.print_help()
        return

    run_processing(config, do_list, do_summary, do_spp, do_multipath)


if __name__ == "__main__":
    main()
