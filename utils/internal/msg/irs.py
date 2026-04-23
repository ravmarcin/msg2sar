# The Infrared Sounder (IRS) functions
import numpy as np
import utils.internal.sentinel.orbit as s1_orb


# ---------------------------------------------------------------------
# 3. Discrete slant delay from refractivity
# ---------------------------------------------------------------------

def slant_delay_from_refractivity(pts_ecef, ds, N_wet_interp):
    """
    Compute wet slant delay (meters) from discrete sampling points.

    pts_ecef: (K,3) ECEF points along LOS (from ground to top)
    ds: (K-1,) segment lengths
    N_wet_interp: function(lat_deg, lon_deg, z_m) -> N_w (N-units)
                  (built from IRS profiles)
    """
    K = pts_ecef.shape[0]
    # Evaluate N at segment midpoints
    N_vals = []
    for k in range(K - 1):
        p_mid = 0.5 * (pts_ecef[k] + pts_ecef[k + 1])
        lat, lon, h = s1_orb.ecef_to_geodetic(*p_mid)
        N_vals.append(N_wet_interp(lat, lon, h))
    N_vals = np.array(N_vals)

    # Discrete approximation of integral L = 1e-6 * sum N(s) ds
    L_w = 1e-6 * np.sum(N_vals * ds)
    return L_w


# ---------------------------------------------------------------------
# 4. Example high-level usage
# ---------------------------------------------------------------------

def example_slant_delay(eof_path, acq_time, tgt_lat, tgt_lon, tgt_h,
                        N_wet_interp, wavelength=0.0555):
    """
    eof_path: path to AUX_*ORB EOF file
    acq_time: datetime of SAR acquisition (UTC)
    tgt_lat, tgt_lon, tgt_h: target geodetic coordinates (deg, deg, m)
    N_wet_interp: function providing N_wet from IRS fields
    wavelength: radar wavelength (m) for phase conversion

    Returns:
        L_w: wet slant delay (m)
        phi_w: corresponding phase delay (rad)
    """
    # Read orbit file and interpolate satellite position
    times, positions = s1_orb.parse_eof_orbit(eof_path)
    r_sat = s1_orb.interp_sat_position(times, positions, acq_time)

    # Target ECEF
    r_tgt = s1_orb.geodetic_to_ecef(tgt_lat, tgt_lon, tgt_h)

    # LOS sampling
    pts_ecef, ds = s1_orb.los_sampling_points(r_tgt, r_sat, z_top=12000.0, dz=100.0)

    # Slant wet delay
    L_w = slant_delay_from_refractivity(pts_ecef, ds, N_wet_interp)

    # Phase delay
    phi_w = 4.0 * np.pi * L_w / wavelength
    return L_w, phi_w