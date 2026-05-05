import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# 1. Read Sentinel-1 EOF orbit file and interpolate satellite position
#    EOF/POD files provide state vectors (position, velocity) in ECEF
#    at 10 s sampling.[web:11][web:14][web:17][web:20]

# ---------------------------------------------------------------------

def parse_eof_orbit(eof_path):
    """
    Parse Sentinel-1 AUX_*ORB EOF file and return times (s since epoch)
    and ECEF positions (m). Assumes standard POD EOF format.[web:11][web:17][web:20]
    """
    tree = ET.parse(eof_path)
    root = tree.getroot()

    # Find Data_Block / List_of_OSVs in EOF XML
    ns = {'eof': root.tag.split('}')[0].strip('{')}
    osv_list = root.find('.//eof:List_of_OSVs', ns)

    times = []
    positions = []

    for osv in osv_list.findall('eof:OSV', ns):
        # UTC time
        t_str = osv.find('eof:UTC', ns).text.strip()
        # Example: "UTC=2018-01-01T12:34:56.000000"
        t_str = t_str.replace('UTC=', '')
        t = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)

        x = float(osv.find('eof:PX', ns).text)  # meters
        y = float(osv.find('eof:PY', ns).text)
        z = float(osv.find('eof:PZ', ns).text)

        times.append(t)
        positions.append([x, y, z])

    times = np.array(times)
    positions = np.array(positions)  # shape (N, 3)
    return times, positions


def interp_sat_position(times, positions, t_query):
    """
    Linear interpolation of satellite ECEF position at time t_query (datetime).
    times: array of datetimes
    positions: array (N,3)
    """
    # Convert times to seconds from first epoch
    t0 = times[0]
    ts = np.array([(t - t0).total_seconds() for t in times])
    tq = (t_query - t0).total_seconds()

    if tq <= ts[0]:
        return positions[0]
    if tq >= ts[-1]:
        return positions[-1]

    # Find bracketing indices
    i2 = np.searchsorted(ts, tq)
    i1 = i2 - 1
    w = (tq - ts[i1]) / (ts[i2] - ts[i1])

    return (1.0 - w) * positions[i1] + w * positions[i2]


# ---------------------------------------------------------------------
# 2. LOS sampling between target and top-of-atmosphere
# ---------------------------------------------------------------------

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """
    Simple WGS84 geodetic -> ECEF conversion.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f**2

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    N = a / np.sqrt(1 - e2 * sin_lat**2)
    x = (N + h_m) * cos_lat * np.cos(lon)
    y = (N + h_m) * cos_lat * np.sin(lon)
    z = (N * (1 - e2) + h_m) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z):
    """
    Approximate ECEF -> geodetic (lat, lon, h) using iterative method.
    Accuracy is sufficient for LOS sampling.[web:18]
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f**2
    b = a * (1 - f)

    r = np.sqrt(x**2 + y**2)
    # Initial latitude
    lat = np.arctan2(z, r * (1 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    h = r / np.cos(lat) - N

    return np.rad2deg(lat), np.rad2deg(lon), h


def los_sampling_points(r_tgt_ecef, r_sat_ecef, z_top=12000.0, dz=100.0):
    """
    Build sampling points along the LOS from ground target to z_top (m above ellipsoid).
    r_tgt_ecef, r_sat_ecef: 3D ECEF vectors (m).
    z_top: top of neutral atmosphere (m).
    dz: approximate vertical step (m).
    Returns:
        pts_ecef: (K,3) ECEF points
        ds: (K-1,) segment lengths
    """
    # Direction from target to satellite
    los_vec = r_sat_ecef - r_tgt_ecef
    los_unit = los_vec / np.linalg.norm(los_vec)

    # Ground height
    lat_t, lon_t, h_t = ecef_to_geodetic(*r_tgt_ecef)

    # Choose heights along LOS from ground up to z_top
    z_levels = np.arange(h_t, z_top + dz, dz)

    pts = []
    for z in z_levels:
        # Move along LOS until geodetic height is (approximately) z
        # Use 1D search along ray
        # Parameter alpha such that r = r_tgt + alpha * los_unit
        # We do a simple bracket + bisection.
        alpha_low = 0.0
        alpha_high = np.linalg.norm(los_vec)  # upper bound
        for _ in range(15):
            alpha_mid = 0.5 * (alpha_low + alpha_high)
            r_mid = r_tgt_ecef + alpha_mid * los_unit
            _, _, h_mid = ecef_to_geodetic(*r_mid)
            if h_mid < z:
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
        r_z = r_tgt_ecef + alpha_high * los_unit
        pts.append(r_z)

    pts = np.array(pts)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, ds