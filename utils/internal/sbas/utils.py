import numpy as np
import xarray as xr
import numpy.ma as ma
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def analyze_modality(data, bandwidth=None, grid_size=1000, prominence=0.01):
    """
    Determine if a distribution is unimodal or multimodal and find modal values.

    Parameters
    ----------
    data : array-like
        1D numeric data
    bandwidth : float or None
        KDE bandwidth. If None, Scott's rule is used.
    grid_size : int
        Number of points to evaluate KDE
    prominence : float
        Minimum prominence of peaks (controls sensitivity)

    Returns
    -------
    result : dict
        {
            'modality': 'unimodal' or 'multimodal',
            'num_modes': int,
            'modes': list of floats
        }
    """

    data = np.asarray(data)

    # KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    x_grid = np.linspace(data.min(), data.max(), grid_size)
    density = kde(x_grid)

    # Find peaks
    peaks, _ = find_peaks(density, prominence=prominence)

    modes = x_grid[peaks]
    num_modes = len(modes)

    modality = "unimodal" if num_modes == 1 else "multimodal"

    return {
        "modality": modality,
        "num_modes": num_modes,
        "modes": modes.tolist()
    }


def shift_minor_modes(data: xr.DataArray) -> xr.DataArray:
    """
    Shift data points associated with minor modes toward the dominant mode.

    This function works on 2D xarray.DataArray inputs and uses a masked array
    approach to handle invalid (NaN) values. Minor modes are identified based
    on the counts of points closest to each mode, and their data is shifted
    so that the smaller modes align with the dominant (largest) mode.
    
    Parameters
    ----------
    data : xr.DataArray
        Input 2D data array. May contain NaN values.
        
    Returns
    -------
    xr.DataArray
        A new DataArray with minor modes shifted toward the dominant mode,
        preserving the original mask and chunking layout.
    """

    # Analyze modality of the finite data points (ignoring NaNs)
    modality = analyze_modality(data.values[np.isfinite(data.values)])
    modes = modality["modes"]

    # Convert data to a masked array, masking NaNs
    d_ma = ma.masked_invalid(data.values)

    # Compute distance from each point to each detected mode
    distances = []
    for mode in modes:
        distances.append(np.abs(d_ma - mode))  # elementwise absolute distance

    # Stack distances and assign each point to the closest mode
    d_gr = np.argmin(distances, axis=0)  # indices of nearest mode for each point

    # Convert assignments to masked array to preserve original mask
    d_gr_ma = ma.masked_array(d_gr, mask=d_ma.mask)

    # Count the number of points assigned to each mode
    ids = list(range(len(modes)))
    count = np.array([np.size(d_gr_ma[d_gr_ma == i]) for i in ids])

    # Identify the dominant mode (the one with the most points)
    mode_big_idx = np.argmax(count)
    mode_big = modes[mode_big_idx]

    # Copy masked data for manipulation
    dn_ma = np.copy(d_ma)

    # Shift points belonging to minor modes toward the dominant mode
    for i, mode in enumerate(modes):
        if i != mode_big_idx:
            mode_dis = mode_big - mode  # shift amount
            idx = np.argwhere(d_gr_ma == i)  # positions of points in this minor mode
            dn_ma[idx[:, 0], idx[:, 1]] += mode_dis  # apply shift

    # Reapply original mask to preserve NaNs
    dn_ma = ma.masked_array(dn_ma, mask=d_ma.mask)

    # Fill masked values with NaN for xarray conversion
    dn_ma_fill = dn_ma.filled(np.nan)

    # Convert the shifted masked array back to xarray.DataArray
    dn_xr = xr.DataArray(dn_ma_fill, dims=("x", "y"))

    # Make a copy of the original DataArray
    data_n = data.copy()

    # Replace values with the shifted data
    data_n.values = dn_xr

    # Preserve original chunking (important for dask-backed arrays)
    data_n_chunk = data_n.chunk(data.data.chunks)

    return data_n_chunk
