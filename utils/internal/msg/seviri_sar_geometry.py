try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import numpy as np
import xarray as xr
from typing import Tuple, Optional, Callable
from utils.internal.log.logger import get_logger
from utils.internal.msg.msg_config import MsgConfig
from utils.internal.sentinel.orbit import (
    los_sampling_points,
    geodetic_to_ecef,
    ecef_to_geodetic,
    interp_sat_position,
    parse_eof_orbit
)

log = get_logger()


class SeviriSarGeometryConverter:
    """
    SEVIRI to SAR geometry converter.

    Converts SEVIRI vertical column measurements to SAR line-of-sight (LOS)
    geometry using satellite orbit data and atmospheric integration.
    """

    def __init__(self, config: MsgConfig):
        self.config = config

        # Get SAR geometry processing config
        self.sar_geom_config = config.msg_processing.get('sar_geometry', {})
        self.z_top_m = self.sar_geom_config.get('z_top_m', 12000)
        self.dz_m = self.sar_geom_config.get('dz_m', 100)
        self.eof_orbit_dir = self.sar_geom_config.get('eof_orbit_dir', 'orbit_files')

        log.info(
            f"SeviriSarGeometryConverter initialized:\n"
            f"  Z top: {self.z_top_m} m\n"
            f"  dZ: {self.dz_m} m\n"
            f"  Orbit dir: {self.eof_orbit_dir}"
        )

    def compute_los_mapping(
        self,
        eof_path: str,
        acquisition_time: str,
        sar_lats: np.ndarray,
        sar_lons: np.ndarray
    ) -> xr.Dataset:
        """
        Compute LOS geometry for each SAR pixel.

        Args:
            eof_path: Path to Sentinel-1 EOF orbit file
            acquisition_time: SAR acquisition time (ISO format string)
            sar_lats: Latitude array (2D or 1D)
            sar_lons: Longitude array (2D or 1D)

        Returns:
            Dataset with los_inc_angle, los_azimuth, los_unit_vectors
        """
        from datetime import datetime

        # Parse orbit file
        times, positions = parse_eof_orbit(eof_path)

        # Get satellite position at acquisition time
        acq_time = datetime.fromisoformat(acquisition_time)
        r_sat_ecef = interp_sat_position(times, positions, acq_time)

        # Create 2D grids if needed
        if sar_lats.ndim == 1 and sar_lons.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(sar_lons, sar_lats)
        else:
            lat_grid = sar_lats
            lon_grid = sar_lons

        # Compute LOS geometry for each pixel
        ny, nx = lat_grid.shape
        los_inc_angles = np.zeros((ny, nx))
        los_azimuths = np.zeros((ny, nx))
        los_unit_vectors = np.zeros((ny, nx, 3))

        log.info(f"Computing LOS geometry for {ny}x{nx} pixels...")

        for i in range(ny):
            for j in range(nx):
                lat = lat_grid[i, j]
                lon = lon_grid[i, j]

                # Target position (assume ground level, h=0)
                r_tgt_ecef = geodetic_to_ecef(lat, lon, 0.0)

                # LOS vector (target to satellite)
                los_vec = r_sat_ecef - r_tgt_ecef
                los_unit = los_vec / np.linalg.norm(los_vec)

                # Compute incidence angle (angle from vertical)
                # Vertical is radial direction from Earth center
                radial = r_tgt_ecef / np.linalg.norm(r_tgt_ecef)
                cos_inc = np.dot(los_unit, radial)
                inc_angle = np.arccos(np.clip(cos_inc, -1, 1))

                # Compute azimuth angle (bearing of satellite from target)
                # East-North-Up local frame
                lat_rad = np.deg2rad(lat)
                lon_rad = np.deg2rad(lon)

                # East, North, Up vectors in ECEF
                east = np.array([
                    -np.sin(lon_rad),
                    np.cos(lon_rad),
                    0
                ])
                north = np.array([
                    -np.sin(lat_rad) * np.cos(lon_rad),
                    -np.sin(lat_rad) * np.sin(lon_rad),
                    np.cos(lat_rad)
                ])

                # Project LOS onto horizontal plane
                los_horiz = los_unit - np.dot(los_unit, radial) * radial
                los_horiz_norm = los_horiz / (np.linalg.norm(los_horiz) + 1e-10)

                # Azimuth from north
                azimuth = np.arctan2(
                    np.dot(los_horiz_norm, east),
                    np.dot(los_horiz_norm, north)
                )

                los_inc_angles[i, j] = inc_angle
                los_azimuths[i, j] = azimuth
                los_unit_vectors[i, j] = los_unit

            if (i + 1) % 10 == 0:
                log.debug(f"Processed {i+1}/{ny} rows")

        # Create xarray Dataset
        dataset = xr.Dataset(
            {
                'los_inc_angle': xr.DataArray(
                    los_inc_angles,
                    dims=['y', 'x'],
                    attrs={'units': 'radians', 'long_name': 'LOS incidence angle'}
                ),
                'los_azimuth': xr.DataArray(
                    los_azimuths,
                    dims=['y', 'x'],
                    attrs={'units': 'radians', 'long_name': 'LOS azimuth angle'}
                ),
                'los_unit_vectors': xr.DataArray(
                    los_unit_vectors,
                    dims=['y', 'x', 'component'],
                    attrs={'long_name': 'LOS unit vectors in ECEF'}
                )
            },
            coords={
                'lat': (['y', 'x'], lat_grid),
                'lon': (['y', 'x'], lon_grid)
            }
        )

        log.info(
            f"Computed LOS geometry:\n"
            f"  Inc angle range: {np.rad2deg(los_inc_angles.min()):.1f}° - "
            f"{np.rad2deg(los_inc_angles.max()):.1f}°\n"
            f"  Azimuth range: {np.rad2deg(los_azimuths.min()):.1f}° - "
            f"{np.rad2deg(los_azimuths.max()):.1f}°"
        )

        return dataset

    def convert_seviri_to_los(
        self,
        seviri_refractivity: xr.DataArray,
        los_geometry: xr.Dataset,
        N_wet_function: Optional[Callable] = None
    ) -> xr.DataArray:
        """
        Project SEVIRI vertical wet delay to SAR LOS.

        Args:
            seviri_refractivity: SEVIRI refractivity data (vertical column)
            los_geometry: LOS geometry dataset from compute_los_mapping
            N_wet_function: Optional function N_wet(lat, lon, z) for integration

        Returns:
            DataArray with SEVIRI delay projected to SAR LOS
        """
        los_inc_angle = los_geometry['los_inc_angle'].values

        # Simple projection: vertical delay / cos(inc_angle)
        # This assumes uniform vertical distribution
        delay_los = seviri_refractivity / np.cos(los_inc_angle)

        # Create output DataArray
        delay_los_da = xr.DataArray(
            delay_los,
            coords=seviri_refractivity.coords,
            dims=seviri_refractivity.dims,
            name='seviri_los_delay',
            attrs={
                'units': 'meters',
                'long_name': 'SEVIRI tropospheric delay in SAR LOS',
                'conversion_method': 'vertical_to_slant'
            }
        )

        log.info(
            f"Converted SEVIRI to LOS:\n"
            f"  Mean vertical delay: {np.nanmean(seviri_refractivity.values):.4f} m\n"
            f"  Mean LOS delay: {np.nanmean(delay_los):.4f} m"
        )

        return delay_los_da

    def slant_delay_from_refractivity(
        self,
        eof_path: str,
        acquisition_time: str,
        target_lat: float,
        target_lon: float,
        N_wet_function: Callable
    ) -> Tuple[float, float]:
        """
        Compute slant path delay using atmospheric integration along LOS.

        This is a more accurate method that integrates refractivity
        along the actual ray path through the atmosphere.

        Args:
            eof_path: Path to EOF orbit file
            acquisition_time: SAR acquisition time
            target_lat: Target latitude
            target_lon: Target longitude
            N_wet_function: Function N_wet(lat, lon, z) returning refractivity

        Returns:
            Tuple of (slant_delay_m, phase_rad)
        """
        from datetime import datetime

        # Parse orbit
        times, positions = parse_eof_orbit(eof_path)
        acq_time = datetime.fromisoformat(acquisition_time)
        r_sat_ecef = interp_sat_position(times, positions, acq_time)

        # Target ECEF position
        r_tgt_ecef = geodetic_to_ecef(target_lat, target_lon, 0.0)

        # Sample along LOS
        pts_ecef, ds = los_sampling_points(
            r_tgt_ecef, r_sat_ecef,
            z_top=self.z_top_m,
            dz=self.dz_m
        )

        # Integrate refractivity
        total_delay = 0.0

        for k in range(len(pts_ecef) - 1):
            # Get midpoint
            mid_pt = 0.5 * (pts_ecef[k] + pts_ecef[k+1])
            lat_k, lon_k, z_k = ecef_to_geodetic(*mid_pt)

            # Evaluate N_wet at this point
            N_wet = N_wet_function(lat_k, lon_k, z_k)

            # Delay contribution: N_wet * 1e-6 * ds
            total_delay += N_wet * 1e-6 * ds[k]

        # Convert to phase (Sentinel-1 C-band)
        wavelength = 0.055465763  # meters
        phase_rad = (4 * np.pi / wavelength) * total_delay

        log.info(
            f"Slant delay at ({target_lat:.3f}, {target_lon:.3f}):\n"
            f"  Delay: {total_delay:.4f} m\n"
            f"  Phase: {phase_rad:.4f} rad"
        )

        return total_delay, phase_rad

    def save_los_geometry(
        self,
        los_geometry: xr.Dataset,
        filename: str = 'seviri_los_geometry.nc'
    ) -> str:
        """
        Save LOS geometry to NetCDF file.

        Args:
            los_geometry: LOS geometry dataset
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = f"{self.config.process_dir}/{filename}"
        los_geometry.to_netcdf(output_path, engine='netcdf4')
        log.info(f"Saved LOS geometry to: {output_path}")
        return output_path

    def load_los_geometry(
        self,
        filename: str = 'seviri_los_geometry.nc'
    ) -> xr.Dataset:
        """
        Load LOS geometry from NetCDF file.

        Args:
            filename: Input filename

        Returns:
            LOS geometry dataset
        """
        input_path = f"{self.config.process_dir}/{filename}"
        los_geometry = xr.open_dataset(input_path, engine='netcdf4')
        log.info(f"Loaded LOS geometry from: {input_path}")
        return los_geometry
