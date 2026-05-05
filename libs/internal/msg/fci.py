import numpy as np
import xarray as xr



# --- Grid index helper (nearest-neighbour for simplicity) ---
def _nearest_index(arr, val):
    idx = np.searchsorted(arr, val) - 1
    if idx < 0:
        idx = 0
    if idx >= len(arr) - 1:
        idx = len(arr) - 2
    return idx


def build_refractivity_from_fci(fci_tcwv_file,
                                fci_tsurf_file=None,
                                z_top=12000.0,
                                H_v_default=2000.0,
                                lapse_rate=0.0065):
    """
    Build functions N_wet_interp(lat, lon, z), N_dry_interp(lat, lon, z)
    from FCI TCWV and (optionally) surface temperature.

    fci_tcwv_file : NetCDF with variables: 'tcwv', 'latitude', 'longitude'
    fci_tsurf_file: NetCDF with near-surface T [K] (optional; uses 288.15 K if None)
    z_top         : top of neutral atmosphere [m] for parameter estimation
    H_v_default   : water vapour scale height [m]
    lapse_rate    : temperature lapse rate [K/m]
    """

    # --- Load FCI data (adapt variable names to your product) ---
    ds_wv = xr.open_dataset(fci_tcwv_file)
    tcwv = ds_wv['tcwv'].values  # kg/m^2
    lats = ds_wv['latitude'].values
    lons = ds_wv['longitude'].values

    if fci_tsurf_file is not None:
        ds_t = xr.open_dataset(fci_tsurf_file)
        T_surf = ds_t['tsurf'].values  # K, same grid as tcwv
    else:
        T_surf = np.full_like(tcwv, 288.15)  # 15 °C default

    # --- Water vapour profile parameters ---
    # Simple exponential profile over [0, z_top]
    H_v = np.full_like(tcwv, H_v_default, dtype=float)  # you can refine this from stability, etc.
    rho0 = 1.2  # near-surface air density [kg/m^3]

    # Integral of q_v(z) * rho(z) dz ≈ q0 * rho0 * H_v * (1 - exp(-z_top / H_v))
    integ_factor = rho0 * H_v * (1.0 - np.exp(-z_top / H_v))
    q0 = tcwv / integ_factor  # surface specific humidity [kg/kg]

    # --- Simple vertical pressure profile (standard atmosphere) ---
    p0 = 101325.0       # Pa
    g = 9.80665         # m/s^2
    R_d = 287.05        # J/(kg K)

    def pressure_std(z):
        # Standard troposphere: P(z) = p0 * (1 - L*z/T0)^(g/(R_d L))
        T0 = 288.15  # K
        L = lapse_rate
        return p0 * (1.0 - L * z / T0) ** (g / (R_d * L))

    # --- Refractivity interpolators ---
    # Microwave refractivity constants (C-band-like)
    k1 = 77.6
    k2 = 71.0
    k3 = 3.75e5

    def N_wet_interp(lat, lon, z):
        # Find nearest FCI pixel
        i_lat = _nearest_index(lats[:, 0] if lats.ndim == 2 else lats, lat)
        i_lon = _nearest_index(lons[0, :] if lons.ndim == 2 else lons, lon)

        # Local parameters
        q0_loc = q0[i_lat, i_lon]
        H_loc = H_v[i_lat, i_lon]
        T0_loc = T_surf[i_lat, i_lon]

        # Profiles
        qv = q0_loc * np.exp(-z / H_loc)     # kg/kg
        T = T0_loc - lapse_rate * z         # K
        T = np.maximum(T, 200.0)            # avoid freezing below ~200 K

        P = pressure_std(z)                 # Pa
        # Approximate vapour pressure e from qv and P
        # qv = 0.622 e / (P - 0.378 e)  -> for small e, e ≈ qv P / 0.622
        e = qv * P / 0.622                  # Pa

        Nw = k2 * e / T + k3 * e / (T ** 2)
        return Nw  # N-units

    def N_dry_interp(lat, lon, z):
        # Dry part is much less variable horizontally, use standard atmosphere
        T0_loc = 288.15
        T = T0_loc - lapse_rate * z
        T = np.maximum(T, 200.0)

        P = pressure_std(z)  # Pa
        # Dry partial pressure ≈ P - e; here e << P so use P directly
        Nd = k1 * P / T
        return Nd

    return N_wet_interp, N_dry_interp
