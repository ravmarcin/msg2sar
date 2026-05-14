try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from libs.internal.sbas.local_setup import local_setup
    local_setup()

import os
import re
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from html.parser import HTMLParser
from libs.internal.log.logger import get_logger

log = get_logger()

# BKG GNSS Data Center endpoints
BASE_URL = "https://igs.bkg.bund.de/root_ftp"

# Available networks on BKG GDC
NETWORKS = [
    "IGS", "EUREF", "MGEX", "GREF", "IGLOS",
    "EPNrepro1", "EPNrepro2", "BfGNet", "IGSac", "MISC", "NTRIP",
]

# RINEX 3 data type suffixes
DATA_TYPES = {
    "MO": "Mixed Observation",
    "GO": "GPS Observation",
    "RO": "GLONASS Observation",
    "EO": "Galileo Observation",
    "CO": "BDS Observation",
    "JO": "QZSS Observation",
    "SO": "SBAS Observation",
    "MN": "Mixed Navigation",
    "GN": "GPS Navigation",
    "RN": "GLONASS Navigation",
    "EN": "Galileo Navigation",
    "CN": "BDS Navigation",
    "JN": "QZSS Navigation",
    "SN": "SBAS Navigation",
    "MM": "Meteorological",
}

# Observation type codes (first character of the 2-char data type)
OBS_TYPES = {"M", "G", "R", "E", "C", "J", "S"}
NAV_CONSTELLATIONS = {"G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BDS", "J": "QZSS", "S": "SBAS", "M": "Mixed"}


class _LinkParser(HTMLParser):
    """Parse href links from an HTML directory listing page."""

    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr_name, attr_value in attrs:
                if attr_name == "href" and attr_value:
                    self.links.append(attr_value)


@dataclass
class RinexFile:
    """Parsed RINEX 3 long filename metadata."""

    station: str        # 4-char station code (e.g. ACOR)
    monument: str       # 2-char monument number (e.g. 00)
    country: str        # 3-char ISO country code (e.g. ESP)
    source: str         # Data source: R=Receiver, S=Stream, U=Unknown
    year: int
    doy: int            # Day of year
    hour: int
    minute: int
    period: str         # e.g. 01D, 01H, 15M
    sampling: str       # Sampling rate (e.g. 30S, 01S) or empty for nav files
    data_type: str      # 2-char code, e.g. MO, GN, EN
    filename: str       # Full filename including extension

    @property
    def is_observation(self) -> bool:
        return len(self.data_type) == 2 and self.data_type[1] == "O"

    @property
    def is_navigation(self) -> bool:
        return len(self.data_type) == 2 and self.data_type[1] == "N"

    @property
    def constellation(self) -> str:
        return NAV_CONSTELLATIONS.get(self.data_type[0], "Unknown")

    @property
    def date(self) -> datetime:
        return datetime(self.year, 1, 1) + timedelta(days=self.doy - 1, hours=self.hour, minutes=self.minute)

    @property
    def description(self) -> str:
        return DATA_TYPES.get(self.data_type, self.data_type)


# RINEX 3 long filename patterns:
#   Nav:  SSSS00CCC_K_YYYYDDDHHMM_PP_TT.rnx.gz
#   Obs:  SSSS00CCC_K_YYYYDDDHHMM_PP_FF_TT.crx.gz  (FF = sampling rate, e.g. 30S)
_RINEX3_PATTERN = re.compile(
    r"^(?P<station>[A-Z0-9]{4})"
    r"(?P<monument>\d{2})"
    r"(?P<country>[A-Z]{3})"
    r"_(?P<source>[RSU])"
    r"_(?P<year>\d{4})"
    r"(?P<doy>\d{3})"
    r"(?P<hour>\d{2})"
    r"(?P<minute>\d{2})"
    r"_(?P<period>\d{2}[DHMS])"
    r"(?:_(?P<sampling>\d{2}[DHMS]))?"
    r"_(?P<data_type>[A-Z]{2})"
    r"\.(?:rnx|crx)"
    r"(?:\.gz)?$"
)


def parse_rinex_filename(filename: str) -> Optional[RinexFile]:
    """Parse a RINEX 3 long filename into its components."""
    match = _RINEX3_PATTERN.match(filename)
    if not match:
        return None

    g = match.groupdict()
    return RinexFile(
        station=g["station"],
        monument=g["monument"],
        country=g["country"],
        source=g["source"],
        year=int(g["year"]),
        doy=int(g["doy"]),
        hour=int(g["hour"]),
        minute=int(g["minute"]),
        period=g["period"],
        sampling=g.get("sampling") or "",
        data_type=g["data_type"],
        filename=filename,
    )


def date_to_doy(date: datetime) -> Tuple[int, int]:
    """Convert a datetime to (year, day-of-year)."""
    doy = date.timetuple().tm_yday
    return date.year, doy


class RinexDownloader:
    """
    GNSS RINEX data downloader for the BKG GNSS Data Center.

    Downloads RINEX 3 observation and navigation files from the BKG GDC archive.
    Data source: https://igs.bkg.bund.de
    Access: anonymous HTTPS (no authentication required).

    URL pattern:
        https://igs.bkg.bund.de/root_ftp/{NETWORK}/obs/{YEAR}/{DOY}/{FILENAME}

    Supports networks: IGS, EUREF, MGEX, GREF, IGLOS, EPNrepro1, EPNrepro2,
    BfGNet, IGSac, MISC, NTRIP.
    """

    def __init__(
        self,
        output_dir: str,
        network: str = "IGS",
        timeout: int = 60,
    ):
        """
        Args:
            output_dir: Directory to save downloaded RINEX files.
            network: BKG network name (e.g. IGS, EUREF, MGEX).
            timeout: HTTP request timeout in seconds.
        """
        if network not in NETWORKS:
            raise ValueError(f"Unknown network '{network}'. Must be one of: {NETWORKS}")

        self.output_dir = output_dir
        self.network = network
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "msg2sar-rinex-downloader/1.0"})

        os.makedirs(self.output_dir, exist_ok=True)
        log.info(f"RinexDownloader initialized: network={network}, output={output_dir}")

    def _build_day_url(self, year: int, doy: int) -> str:
        """Build the URL for a specific day's directory listing."""
        return f"{BASE_URL}/{self.network}/obs/{year}/{doy:03d}/"

    def list_files(
        self,
        date: datetime,
        station: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> List[RinexFile]:
        """
        List available RINEX files for a given date.

        Args:
            date: Date to list files for.
            station: Optional 4-char station filter (case-insensitive).
            data_type: Optional 2-char data type filter (e.g. 'MO', 'GN').

        Returns:
            List of parsed RinexFile objects matching the filters.
        """
        year, doy = date_to_doy(date)
        url = self._build_day_url(year, doy)

        log.info(f"Listing RINEX files: {url}")

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                log.warning(f"No data directory found for {year}/{doy:03d} on {self.network}")
                return []
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to list files: {e}")
            return []

        # Parse links from HTML directory listing
        parser = _LinkParser()
        parser.feed(resp.text)

        results = []
        for link in parser.links:
            # Extract just the filename from the link
            fname = link.rstrip("/").split("/")[-1]
            parsed = parse_rinex_filename(fname)
            if parsed is None:
                continue

            if station and parsed.station.upper() != station.upper():
                continue
            if data_type and parsed.data_type != data_type.upper():
                continue

            results.append(parsed)

        log.info(f"Found {len(results)} RINEX files for {year}/{doy:03d}")
        return results

    def download_file(
        self,
        date: datetime,
        filename: str,
        overwrite: bool = False,
    ) -> Optional[str]:
        """
        Download a single RINEX file by name.

        Args:
            date: Date the file belongs to (for constructing the URL path).
            filename: Full RINEX filename (e.g. ACOR00ESP_R_20200040000_01D_MO.rnx.gz).
            overwrite: If True, re-download even if the file already exists locally.

        Returns:
            Local file path on success, None on failure.
        """
        year, doy = date_to_doy(date)
        url = f"{BASE_URL}/{self.network}/obs/{year}/{doy:03d}/{filename}"
        local_path = os.path.join(self.output_dir, filename)

        if not overwrite and os.path.exists(local_path):
            log.info(f"Already exists, skipping: {filename}")
            return local_path

        log.info(f"Downloading: {filename}")

        try:
            resp = self._session.get(url, timeout=self.timeout, stream=True)
            resp.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = os.path.getsize(local_path) / 1024
            log.info(f"  Saved: {local_path} ({size_kb:.1f} KB)")
            return local_path

        except requests.exceptions.HTTPError:
            log.warning(f"  File not found on server: {filename} (HTTP {resp.status_code})")
            return None
        except requests.exceptions.RequestException as e:
            log.error(f"  Download failed: {e}")
            # Clean up partial file
            if os.path.exists(local_path):
                os.remove(local_path)
            return None

    def download_station(
        self,
        station: str,
        date: datetime,
        obs: bool = True,
        nav: bool = False,
        constellation: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Download all RINEX files for a station on a given date.

        Args:
            station: 4-char station code (e.g. 'ACOR').
            date: Date to download for.
            obs: If True, download observation files.
            nav: If True, download navigation files.
            constellation: Optional constellation filter
                           ('G'=GPS, 'R'=GLONASS, 'E'=Galileo, 'C'=BDS, 'M'=Mixed).
            overwrite: If True, re-download existing files.

        Returns:
            List of local file paths for successfully downloaded files.
        """
        available = self.list_files(date, station=station)

        if not available:
            log.warning(f"No files found for station {station} on {date.strftime('%Y-%m-%d')}")
            return []

        # Filter by observation/navigation
        filtered = []
        for rf in available:
            if rf.is_observation and not obs:
                continue
            if rf.is_navigation and not nav:
                continue
            if constellation and rf.data_type[0] != constellation.upper():
                continue
            filtered.append(rf)

        if not filtered:
            log.warning(f"No matching files after filtering (obs={obs}, nav={nav}, constellation={constellation})")
            return []

        log.info(
            f"Downloading {len(filtered)} file(s) for {station} on {date.strftime('%Y-%m-%d')} "
            f"from {self.network}"
        )

        downloaded = []
        for rf in filtered:
            path = self.download_file(date, rf.filename, overwrite=overwrite)
            if path:
                downloaded.append(path)

        log.info(f"Downloaded {len(downloaded)}/{len(filtered)} files for {station}")
        return downloaded

    def download_date_range(
        self,
        station: str,
        start_date: datetime,
        end_date: datetime,
        obs: bool = True,
        nav: bool = False,
        constellation: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Download RINEX files for a station over a date range.

        Args:
            station: 4-char station code.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            obs: If True, download observation files.
            nav: If True, download navigation files.
            constellation: Optional constellation filter.
            overwrite: If True, re-download existing files.

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to lists of downloaded file paths.
        """
        results = {}
        current = start_date

        total_days = (end_date - start_date).days + 1
        log.info(
            f"Downloading RINEX for {station} from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')} ({total_days} days)"
        )

        while current <= end_date:
            date_key = current.strftime("%Y-%m-%d")
            paths = self.download_station(
                station=station,
                date=current,
                obs=obs,
                nav=nav,
                constellation=constellation,
                overwrite=overwrite,
            )
            if paths:
                results[date_key] = paths
            current += timedelta(days=1)

        total_files = sum(len(v) for v in results.values())
        log.info(f"Completed: {total_files} files over {len(results)}/{total_days} days")
        return results

    def download_stations(
        self,
        stations: List[str],
        date: datetime,
        obs: bool = True,
        nav: bool = False,
        constellation: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Download RINEX files for multiple stations on a single date.

        Args:
            stations: List of 4-char station codes.
            date: Date to download for.
            obs: If True, download observation files.
            nav: If True, download navigation files.
            constellation: Optional constellation filter.
            overwrite: If True, re-download existing files.

        Returns:
            Dict mapping station codes to lists of downloaded file paths.
        """
        results = {}

        log.info(f"Downloading RINEX for {len(stations)} stations on {date.strftime('%Y-%m-%d')}")

        for station in stations:
            paths = self.download_station(
                station=station,
                date=date,
                obs=obs,
                nav=nav,
                constellation=constellation,
                overwrite=overwrite,
            )
            if paths:
                results[station.upper()] = paths

        total_files = sum(len(v) for v in results.values())
        log.info(f"Completed: {total_files} files for {len(results)}/{len(stations)} stations")
        return results
