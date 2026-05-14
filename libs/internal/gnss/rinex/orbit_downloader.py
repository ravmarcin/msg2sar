"""
Precise orbit (SP3) and clock file downloader for GNSS processing.

Downloads IGS final products from the GFZ data center:
    ftp://ftp.gfz-potsdam.de/pub/GNSS/products/final/wWWWW/

Files needed for sp3_interp (Lagrange orbit interpolation):
    - SP3: yesterday, today, tomorrow (3 files)
    - CLK: today (1 file)

Filename format: {product}{gps_week}{weekday}.{sp3|clk}.Z
    e.g. gfz21526.sp3.Z = GFZ final orbit, GPS week 2152, Saturday (day 6)
"""

import os
import urllib.request
import subprocess
from datetime import date, timedelta
from typing import List, Tuple

from libs.internal.log.logger import get_logger

log = get_logger()

GFZ_BASE_URL = "ftp://ftp.gfz-potsdam.de/pub/GNSS/products/final"


def _gps_week_day(epoch: date) -> Tuple[int, int]:
    """Compute GPS week and weekday (0=Sunday) for a given date."""
    gps_epoch = date(1980, 1, 6)
    delta = (epoch - gps_epoch).days
    gps_week = delta // 7
    gps_weekday = delta % 7
    return gps_week, gps_weekday


def _sp3_filename(epoch: date, product: str = "gfz") -> str:
    """Generate SP3 filename for a given date."""
    gps_week, gps_weekday = _gps_week_day(epoch)
    return f"{product}{gps_week}{gps_weekday}.sp3"


def _clk_filename(epoch: date, product: str = "gfz") -> str:
    """Generate clock filename for a given date."""
    gps_week, gps_weekday = _gps_week_day(epoch)
    return f"{product}{gps_week}{gps_weekday}.clk"


def _download_file(url: str, output_path: str, overwrite: bool = False) -> bool:
    """Download a file from FTP URL."""
    if os.path.exists(output_path) and not overwrite:
        log.info(f"  Already exists: {os.path.basename(output_path)}")
        return True

    try:
        log.info(f"  Downloading: {os.path.basename(url)}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        log.warning(f"  Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def _decompress_z(z_path: str) -> str:
    """Decompress a .Z (Unix compress) file."""
    out_path = z_path[:-2]  # Remove .Z
    if os.path.exists(out_path):
        return out_path

    try:
        # Try using uncompress / gzip
        subprocess.run(
            ["uncompress", "-f", z_path],
            check=True, capture_output=True
        )
        if os.path.exists(out_path):
            return out_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: try gzip (handles .Z files too)
    try:
        subprocess.run(
            ["gzip", "-df", z_path],
            check=True, capture_output=True
        )
        if os.path.exists(out_path):
            return out_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: try ncompress Python package
    try:
        import ncompress
        with open(z_path, "rb") as f_in:
            data = ncompress.decompress(f_in.read())
        with open(out_path, "wb") as f_out:
            f_out.write(data)
        os.remove(z_path)
        return out_path
    except (ImportError, Exception) as e:
        log.error(f"  Failed to decompress {z_path}: {e}")
        return z_path


def download_orbits(
    epoch: date,
    output_dir: str,
    product: str = "gfz",
    overwrite: bool = False,
) -> dict:
    """
    Download SP3 orbit and clock files needed for orbit interpolation.

    Downloads 3 SP3 files (yesterday, today, tomorrow) and 1 clock file (today)
    from the GFZ data center.

    Args:
        epoch: Date to process.
        output_dir: Directory to save files.
        product: Product source (default: 'gfz').
        overwrite: Re-download existing files.

    Returns:
        Dict with keys 'sp3_yesterday', 'sp3_today', 'sp3_tomorrow', 'clk_today'
        mapping to local file paths. Missing files map to None.
    """
    os.makedirs(output_dir, exist_ok=True)

    yesterday = epoch - timedelta(days=1)
    tomorrow = epoch + timedelta(days=1)

    files_needed = {
        "sp3_yesterday": _sp3_filename(yesterday, product),
        "sp3_today": _sp3_filename(epoch, product),
        "sp3_tomorrow": _sp3_filename(tomorrow, product),
        "clk_today": _clk_filename(epoch, product),
    }

    log.info(f"Downloading orbit products for {epoch} (product={product})")

    results = {}
    for key, filename in files_needed.items():
        # Determine GPS week for the URL path
        file_epoch = yesterday if "yesterday" in key else (tomorrow if "tomorrow" in key else epoch)
        gps_week, _ = _gps_week_day(file_epoch)

        compressed_name = f"{filename}.Z"
        url = f"{GFZ_BASE_URL}/w{gps_week}/{compressed_name}"
        compressed_path = os.path.join(output_dir, compressed_name)
        final_path = os.path.join(output_dir, filename)

        # If decompressed file already exists, skip
        if os.path.exists(final_path) and not overwrite:
            log.info(f"  Already exists: {filename}")
            results[key] = final_path
            continue

        # Download compressed file
        if _download_file(url, compressed_path, overwrite):
            decompressed = _decompress_z(compressed_path)
            if os.path.exists(decompressed):
                results[key] = decompressed
                log.info(f"  Ready: {os.path.basename(decompressed)}")
            else:
                results[key] = None
                log.error(f"  Decompression failed: {compressed_name}")
        else:
            results[key] = None

    n_ok = sum(1 for v in results.values() if v is not None)
    log.info(f"Orbit files ready: {n_ok}/{len(files_needed)}")

    return results


def download_orbits_for_range(
    start_date: date,
    end_date: date,
    output_dir: str,
    product: str = "gfz",
    overwrite: bool = False,
) -> List[str]:
    """
    Download all orbit files needed for a date range.

    Args:
        start_date: First date.
        end_date: Last date (inclusive).
        output_dir: Directory to save files.
        product: Product source.
        overwrite: Re-download existing.

    Returns:
        List of all downloaded file paths.
    """
    all_files = []
    current = start_date
    while current <= end_date:
        result = download_orbits(current, output_dir, product, overwrite)
        all_files.extend(v for v in result.values() if v is not None)
        current += timedelta(days=1)

    # Deduplicate (adjacent days share SP3 files)
    return sorted(set(all_files))
