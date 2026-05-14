try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from libs.internal.sbas.local_setup import local_setup
    local_setup()

import os
import gzip
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from libs.internal.log.logger import get_logger

log = get_logger()


class GnssRinexProcessor:
    """
    GNSS RINEX data processor using the gnsspy library.

    Reads RINEX 3 observation and navigation files downloaded by RinexDownloader,
    and provides methods for:
    - Reading observation and navigation data
    - Computing satellite positions via SP3 orbit interpolation
    - Single Point Positioning (SPP)
    - Multipath analysis
    - Tropospheric delay estimation

    Requires: gnsspy (pip install gnsspy)
    Reference: https://github.com/GNSSpy-Project/gnsspy
    """

    def __init__(self, rinex_dir: str):
        """
        Args:
            rinex_dir: Directory containing downloaded RINEX files (.rnx.gz / .crx.gz).
        """
        self.rinex_dir = rinex_dir
        self._observations = {}
        self._navigations = {}

        if not os.path.isdir(rinex_dir):
            raise FileNotFoundError(f"RINEX directory not found: {rinex_dir}")

        log.info(f"GnssRinexProcessor initialized: {rinex_dir}")

    @staticmethod
    def _decompress_gz(gz_path: str) -> str:
        """Decompress a .gz file in-place and return the decompressed path."""
        if not gz_path.endswith(".gz"):
            return gz_path

        out_path = gz_path[:-3]
        if os.path.exists(out_path):
            return out_path

        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        log.info(f"Decompressed: {os.path.basename(gz_path)}")
        return out_path

    @staticmethod
    def _crx_to_rnx(crx_path: str) -> str:
        """Convert Hatanaka compressed RINEX (.crx) to standard RINEX (.rnx)."""
        if not crx_path.endswith(".crx"):
            return crx_path

        rnx_path = crx_path.replace(".crx", ".rnx")
        if os.path.exists(rnx_path):
            return rnx_path

        # Use hatanaka package (cross-platform, pure Python wrapper)
        try:
            import hatanaka
            with open(crx_path, "rb") as f:
                crx_data = f.read()
            rnx_data = hatanaka.decompress(crx_data)
            with open(rnx_path, "wb") as f:
                f.write(rnx_data)
            log.info(f"Converted CRX -> RNX: {os.path.basename(rnx_path)}")
            return rnx_path
        except ImportError:
            pass
        except Exception as e:
            log.warning(f"hatanaka CRX to RNX conversion failed: {e}")

        # Fallback to gnsspy bundled binary
        try:
            import gnsspy
            gnsspy.crx2rnx(crx_path)
            if os.path.exists(rnx_path):
                log.info(f"Converted CRX -> RNX: {os.path.basename(rnx_path)}")
                return rnx_path
            else:
                log.warning(f"crx2rnx did not produce expected output: {rnx_path}")
                return crx_path
        except Exception as e:
            log.warning(f"CRX to RNX conversion failed: {e}")
            return crx_path

    def _prepare_file(self, filename: str) -> str:
        """Decompress .gz and convert .crx to .rnx as needed."""
        filepath = os.path.join(self.rinex_dir, filename)

        if filepath.endswith(".gz"):
            filepath = self._decompress_gz(filepath)

        if filepath.endswith(".crx"):
            filepath = self._crx_to_rnx(filepath)

        return filepath

    def read_observation(self, obs_filename: str):
        """
        Read a RINEX observation file.

        Args:
            obs_filename: Observation filename (e.g. BOGO00POL_R_20210100000_01D_30S_MO.crx.gz).

        Returns:
            gnsspy Observation object with attributes:
                - observation: DataFrame (MultiIndex: Epoch, SV) with pseudorange, carrier phase, etc.
                - approx_position: [X, Y, Z] in ECEF
                - epoch: date of the observation file
                - interval: sampling interval in seconds
                - receiver_type, antenna_type, version
        """
        import gnsspy

        if obs_filename in self._observations:
            return self._observations[obs_filename]

        filepath = self._prepare_file(obs_filename)
        log.info(f"Reading observation file: {os.path.basename(filepath)}")

        obs = gnsspy.read_obsFile(filepath)
        self._observations[obs_filename] = obs

        n_epochs = obs.observation.index.get_level_values("Epoch").nunique()
        n_svs = obs.observation.index.get_level_values("SV").nunique()
        log.info(
            f"  Loaded: {n_epochs} epochs, {n_svs} satellites, "
            f"interval={obs.interval}s, version={obs.version}"
        )

        return obs

    def read_navigation(self, nav_filename: str):
        """
        Read a RINEX navigation file.

        Args:
            nav_filename: Navigation filename (e.g. BOGO00POL_R_20210100000_01D_GN.rnx.gz).

        Returns:
            Navigation DataFrame with broadcast ephemeris data.
        """
        import gnsspy

        if nav_filename in self._navigations:
            return self._navigations[nav_filename]

        filepath = self._prepare_file(nav_filename)
        log.info(f"Reading navigation file: {os.path.basename(filepath)}")

        nav = gnsspy.read_navFile(filepath)
        self._navigations[nav_filename] = nav

        log.info(f"  Loaded navigation data: {len(nav)} ephemeris records")

        return nav

    def compute_spp(
        self,
        obs_filename: str,
        orbit,
        system: str = "G",
        cut_off: float = 7.0,
    ) -> pd.DataFrame:
        """
        Compute Single Point Positioning (SPP) using ionosphere-free combination.

        Args:
            obs_filename: Observation filename.
            orbit: Interpolated orbit DataFrame (from sp3_interp or read_sp3File).
            system: GNSS system code ('G'=GPS, 'R'=GLONASS, 'E'=Galileo, 'G+R+E' for multi).
            cut_off: Elevation cutoff angle in degrees.

        Returns:
            DataFrame with columns: Epoch, X, Y, Z, receiver_clock, Latitude, Longitude, Height.
        """
        import gnsspy

        station = self.read_observation(obs_filename)
        log.info(f"Computing SPP for {obs_filename} (system={system}, cutoff={cut_off})")

        position = gnsspy.spp(station, orbit, system=system, cut_off=cut_off)

        log.info(f"  SPP computed: {len(position)} epoch solutions")
        return position

    def compute_multipath(
        self,
        obs_filename: str,
        system: str = "G",
    ) -> pd.DataFrame:
        """
        Compute multipath indicators (MP1, MP2) from dual-frequency observations.

        Args:
            obs_filename: Observation filename.
            system: GNSS system code.

        Returns:
            DataFrame with Multipath1 and Multipath2 columns per satellite per epoch.
        """
        import gnsspy

        station = self.read_observation(obs_filename)
        log.info(f"Computing multipath for {obs_filename} (system={system})")

        mp = gnsspy.multipath(station, system=system)

        log.info(f"  Multipath computed: {len(mp)} records")
        return mp

    def compute_tropospheric_delay(
        self,
        obs_filename: str,
        elevation: float = 90.0,
    ) -> float:
        """
        Compute zenith tropospheric delay at the station location using Collins (1999) model.

        Uses the station's approximate ECEF position from the RINEX header.

        Args:
            obs_filename: Observation filename.
            elevation: Elevation angle in degrees (default 90 = zenith).

        Returns:
            Tropospheric delay in meters.
        """
        import gnsspy

        station = self.read_observation(obs_filename)
        x, y, z = station.approx_position
        epoch = station.epoch

        delay = gnsspy.tropospheric_delay(x, y, z, elevation, epoch)

        log.info(
            f"  Tropospheric delay at zenith: {delay:.4f} m "
            f"(station approx pos: [{x:.1f}, {y:.1f}, {z:.1f}])"
        )
        return delay

    def interpolate_orbits(
        self,
        epoch: datetime,
        interval: int = 30,
        sp3_product: str = "gfz",
        clock_product: str = "gfz",
    ):
        """
        Interpolate precise satellite orbits from SP3 files using Lagrange interpolation.

        SP3 and clock files must be pre-downloaded into the working directory.

        Args:
            epoch: Date for orbit interpolation.
            interval: Clock interpolation interval in seconds.
            sp3_product: SP3 product source ('gfz', 'igs', 'cod', etc.).
            clock_product: Clock product source.

        Returns:
            Interpolated orbit DataFrame with columns: X, Y, Z, Vx, Vy, Vz, deltaT.
        """
        import gnsspy

        log.info(f"Interpolating orbits for {epoch.strftime('%Y-%m-%d')} (product={sp3_product})")

        orbit = gnsspy.sp3_interp(
            epoch, interval=interval,
            sp3_product=sp3_product, clock_product=clock_product,
        )

        n_svs = orbit.index.get_level_values("SV").nunique()
        n_epochs = orbit.index.get_level_values("Epoch").nunique()
        log.info(f"  Orbit interpolated: {n_svs} satellites, {n_epochs} epochs")

        return orbit

    def compute_spp_with_orbits(
        self,
        obs_filename: str,
        orbit_dir: str,
        system: str = "G",
        cut_off: float = 7.0,
        sp3_product: str = "gfz",
        clock_product: str = "gfz",
        interval: int = 30,
    ) -> pd.DataFrame:
        """
        Run full SPP pipeline: interpolate orbits + compute per-epoch positions.

        Returns per-epoch (30s) positions with observation-derived tropospheric
        correction using actual satellite elevation angles.

        Args:
            obs_filename: Observation filename.
            orbit_dir: Directory containing SP3 and clock files.
            system: GNSS system code ('G'=GPS).
            cut_off: Elevation cutoff angle in degrees.
            sp3_product: SP3 product prefix (e.g. 'gfz', 'igs').
            clock_product: Clock product prefix.
            interval: Orbit interpolation interval in seconds.

        Returns:
            DataFrame indexed by Epoch with columns: X, Y, Z, receiver_clock,
            n_sats, tropo_mean_m.
        """
        import gnsspy
        from gnsspy.position.position import gnssDataframe, _observation_picker
        from gnsspy.position.satellite import _sagnac, _azel, _reception_coord
        from gnsspy.geodesy.coordinate import _distance_euclidean
        from gnsspy.funcs.constants import _CLIGHT
        from datetime import timedelta

        station = self.read_observation(obs_filename)
        epoch = station.epoch

        # Fix gnsspy bug: receiver_clock parsed as string from RINEX header
        if isinstance(station.receiver_clock, str):
            station.receiver_clock = float(station.receiver_clock)

        # gnsspy expects SP3/clock files in the working directory
        original_cwd = os.getcwd()
        try:
            os.chdir(orbit_dir)
            log.info(f"Interpolating orbits from {orbit_dir} (product={sp3_product})")
            orbit = gnsspy.sp3_interp(
                epoch, interval=interval,
                sp3_product=sp3_product, clock_product=clock_product,
            )
            n_svs = orbit.index.get_level_values("SV").nunique()
            log.info(f"  Orbit interpolated: {n_svs} satellites")
        finally:
            os.chdir(original_cwd)

        # Build GNSS dataframe (merges obs + orbit + applies elevation cutoff)
        log.info(f"Computing per-epoch SPP (system={system}, cutoff={cut_off})")
        observation_list = _observation_picker(station, system)
        gnss = gnssDataframe(station, orbit, system, cut_off)

        if len(observation_list) < 2:
            log.error("  Ionosphere-free combination not available (need dual-frequency)")
            return pd.DataFrame()

        # Ionosphere-free pseudorange combination
        pseudorange1 = getattr(gnss, observation_list[0][3])
        pseudorange2 = getattr(gnss, observation_list[1][3])
        frequency1 = observation_list[0][4]
        frequency2 = observation_list[1][4]

        gnss["Ionosphere_Free"] = (
            (frequency1**2 * pseudorange1 - frequency2**2 * pseudorange2)
            / (frequency1**2 - frequency2**2)
        )
        gnss = gnss.dropna(subset=["Ionosphere_Free"])
        gnss["Travel_time"] = gnss["Ionosphere_Free"] / _CLIGHT
        gnss["X_Reception"], gnss["Y_Reception"], gnss["Z_Reception"] = _reception_coord(
            gnss.X, gnss.Y, gnss.Z, gnss.Vx, gnss.Vy, gnss.Vz, gnss.Travel_time
        )

        # Epoch-by-epoch least squares
        epoch_list = gnss.index.get_level_values("Epoch").unique().sort_values()
        approx_position = list(station.approx_position)
        receiver_clock = station.receiver_clock

        position_list = []
        epoch_interval = timedelta(seconds=station.interval - 0.000001)

        for ep in epoch_list:
            ep_end = ep + epoch_interval
            gnss_temp = gnss.loc[ep:ep_end].copy()

            if len(gnss_temp) < 4:
                continue

            # Iterative solution (6 iterations per epoch)
            for _ in range(6):
                distance = _distance_euclidean(
                    approx_position[0], approx_position[1], approx_position[2],
                    gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception,
                )
                gnss_temp["Distance"] = distance + _sagnac(
                    approx_position[0], approx_position[1], approx_position[2],
                    gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception,
                )
                gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(
                    station.approx_position[0], station.approx_position[1], station.approx_position[2],
                    gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance,
                )
                gnss_temp["Tropo"] = gnsspy.tropospheric_delay(
                    station.approx_position[0], station.approx_position[1], station.approx_position[2],
                    gnss_temp.Elevation, station.epoch,
                )

                coeff = np.zeros([len(gnss_temp), 4])
                coeff[:, 0] = (approx_position[0] - gnss_temp.X_Reception) / gnss_temp.Distance
                coeff[:, 1] = (approx_position[1] - gnss_temp.Y_Reception) / gnss_temp.Distance
                coeff[:, 2] = (approx_position[2] - gnss_temp.Z_Reception) / gnss_temp.Distance
                coeff[:, 3] = 1

                l_matrix = (
                    gnss_temp.Ionosphere_Free - gnss_temp.Distance
                    + _CLIGHT * (gnss_temp.DeltaTSV + gnss_temp.Relativistic_clock - receiver_clock)
                    - gnss_temp.Tropo
                )

                try:
                    solution = np.linalg.lstsq(coeff, np.array(l_matrix), rcond=None)
                    dx = solution[0]
                    approx_position[0] += dx[0]
                    approx_position[1] += dx[1]
                    approx_position[2] += dx[2]
                    receiver_clock += dx[3] / _CLIGHT
                except np.linalg.LinAlgError:
                    break

            position_list.append({
                "Epoch": ep,
                "X": approx_position[0],
                "Y": approx_position[1],
                "Z": approx_position[2],
                "receiver_clock": receiver_clock,
                "n_sats": len(gnss_temp),
                "tropo_mean_m": float(gnss_temp["Tropo"].mean()),
            })

        df = pd.DataFrame(position_list)
        if not df.empty:
            df = df.set_index("Epoch")

        log.info(f"  SPP computed: {len(df)} epoch solutions")
        return df

    def get_station_position(self, obs_filename: str) -> Dict[str, float]:
        """
        Get the approximate station position from the RINEX observation header.

        Args:
            obs_filename: Observation filename.

        Returns:
            Dict with keys: X, Y, Z (ECEF meters), lat, lon, height (WGS84 degrees/meters).
        """
        station = self.read_observation(obs_filename)
        x, y, z = station.approx_position

        # ECEF to geodetic (WGS84)
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f ** 2
        lon = np.degrees(np.arctan2(y, x))
        p = np.sqrt(x ** 2 + y ** 2)
        lat = np.degrees(np.arctan2(z, p * (1 - e2)))

        # Iterative latitude
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat)) ** 2)
            lat = np.degrees(np.arctan2(z + e2 * N * np.sin(np.radians(lat)), p))

        N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat)) ** 2)
        height = p / np.cos(np.radians(lat)) - N

        return {
            "X": x, "Y": y, "Z": z,
            "lat": lat, "lon": lon, "height": height,
        }

    def get_observation_summary(self, obs_filename: str) -> Dict:
        """
        Get a summary of an observation file.

        Args:
            obs_filename: Observation filename.

        Returns:
            Dict with file metadata and observation statistics.
        """
        station = self.read_observation(obs_filename)
        obs = station.observation

        epochs = obs.index.get_level_values("Epoch").unique()
        svs = obs.index.get_level_values("SV").unique()
        systems = obs["SYSTEM"].unique().tolist()

        return {
            "filename": station.filename,
            "epoch": str(station.epoch),
            "version": station.version,
            "receiver": station.receiver_type,
            "antenna": station.antenna_type,
            "interval_s": station.interval,
            "n_epochs": len(epochs),
            "time_start": str(epochs.min()),
            "time_end": str(epochs.max()),
            "n_satellites": len(svs),
            "systems": systems,
            "observation_types": station.observation_types,
            "approx_position": list(station.approx_position),
        }

    def list_rinex_files(self) -> Dict[str, List[str]]:
        """
        List RINEX files in the data directory, grouped by type.

        Returns:
            Dict with keys 'observation', 'navigation', 'other' mapping to filename lists.
        """
        from libs.internal.gnss.rinex.rinex_downloader import parse_rinex_filename

        result = {"observation": [], "navigation": [], "other": []}

        for fname in sorted(os.listdir(self.rinex_dir)):
            parsed = parse_rinex_filename(fname.replace(".gz", "").replace(".crx", ".rnx"))
            if parsed is None:
                parsed = parse_rinex_filename(fname)

            if parsed and parsed.is_observation:
                result["observation"].append(fname)
            elif parsed and parsed.is_navigation:
                result["navigation"].append(fname)
            else:
                if fname.endswith((".rnx", ".rnx.gz", ".crx", ".crx.gz")):
                    result["other"].append(fname)

        log.info(
            f"RINEX files: {len(result['observation'])} obs, "
            f"{len(result['navigation'])} nav, {len(result['other'])} other"
        )
        return result
