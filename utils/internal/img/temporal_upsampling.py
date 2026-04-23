import numpy as np
from utils.internal.img.utils import add_x_end, add_y_end, add_x_start, add_y_start


def time_trans(dx_ref, dy_ref, dt_ref, dt_fit, img, nd_val=np.nan):
    """
    Function to translate the image based on the reference translation
    :param dx_ref: int                  - x-axis reference translation
    :param dy_ref: int                  - y-axis reference translation
    :param dt_ref: datetime.timedelta   - reference time difference
    :param dt_fit: datetime.timedelta   - time difference to fit
    :param img: 2d np.array             - reference image to translate
    :param nd_val: float                - value to fill for no data pixels
    :return img_t: 2d np.array          - translated image
    :return: tuple                      - computed x and y translation of a new image
    """
    if dt_ref != 0:
        ratio_ = dt_fit / dt_ref
    else:
        ratio_ = 0
    dx_fit, dy_fit = (int(dx_ref * ratio_), int(dy_ref * ratio_))
    if dx_fit == 0 and dy_fit == 0:
        return img, (dx_fit, dy_fit)
    else:
        img_t = img.copy()
        if dy_fit != 0:
            img_t = np.roll(img_t, dy_fit, axis=0)
            if dy_fit > 0:
                img_t[-dy_fit:, :] = nd_val
            else:
                img_t[:-dy_fit, :] = nd_val
        if dx_fit != 0:
            img_t = np.roll(img_t, dx_fit, axis=1)
            if dx_fit > 0:
                img_t[:, -dx_fit:] = nd_val
            else:
                img_t[:, :-dx_fit] = nd_val
        return img_t, (dx_fit, dy_fit)


def temporal_upsampling(arr_m, arr_s, m_dx, m_dy, sev_m_sar_dif, sev_s_sar_dif, nd_val):
    """
    Temporal upsampling ofthe images based on the:
    - translation parameters (space vector) and 
    - time differences (time vector)
    :param arr_m: 2d np.array                   - master image
    :param arr_s: 2d np.array                   - slave image
    :param m_dx: int                            - x translation
    :param m_dy: int                            - y translation
    :param sev_m_sar_dif: datetime.timedelta    - time difference between reference image and master image
    :param sev_s_sar_dif: datetime.timedelta    - time difference between reference image and slave image
    :param nd_val: float                        - value to fill for no data pixels
    :return: 2d np.array                        - upsampled image
    """

    # Get the total time difference between the two images (master and slave)
    dt_ref = sev_m_sar_dif + sev_s_sar_dif

    # Translate (shift) both images to the reference time
    img_t1, _ = time_trans(
        dx_ref=m_dx,
        dy_ref=m_dy,
        dt_ref=dt_ref,
        dt_fit=sev_m_sar_dif,
        img=arr_m,
        nd_val=nd_val
    )

    img_t2, _ = time_trans(
        dx_ref=-m_dx,
        dy_ref=-m_dy,
        dt_ref=dt_ref,
        dt_fit=sev_s_sar_dif,
        img=arr_s,
        nd_val=nd_val
    )

    # Average the two translated images to get the upsampled image at the reference time
    img_n = (img_t1 + img_t2) / 2

    return img_n



def img_match_err(arr_ref, arr_fit, max_i, max_j, cut_off, min_i=0, min_j=0, iter_=1, title=' Matching'):
    """
    Function to generate to root square errors of the image matching
    :param arr_ref: 2d np.array - reference image
    :param arr_fit: 2d np.array - image to fit
    :param max_i: int           - max iteration size of i axis
    :param max_j: int           - max iteration size of i axis
    :param cut_off: int         - image cut-off parameter
    :param min_i: int           - min iteration size of j axis
    :param min_j: int           - min iteration size of j axis
    :param iter_: int           - iteration interval
    :param title: str           - processing bar title
    :return err_l: list         - list of errors and assigned i-th and j-th index
    """
    n_ = np.shape(arr_ref)[0] * np.shape(arr_ref)[1]
    err_l = []
    for i in range(min_i, max_i, iter_):
        for j in range(min_j, max_j, iter_):
            arr_fit_ij = arr_fit[i:-(cut_off - i), j:-(cut_off - j)]
            err_ij = np.sum(np.abs(arr_fit_ij - arr_ref)) / n_
            err_l.append((i, j, err_ij))
    return err_l


def img_match_trans(arr_ref, arr_fit, steps_, cut_ref=350, cut_fit=115):
    """
    Function to identify the translation parameters (dx, dy) for a pair of dt seperated images
    translation: image_fit(t_i) -> image_ref(t_j)
    :param arr_ref: 2d np.array - reference image
    :param arr_fit: 2d np.array - image to fit
    :param steps_: list/tuple   - list of steps for (intervals in a for loop)
    :param cut_ref: int         - cut size for one side for reference image
    :param cut_fit: int         - cut size for one side for image to fit
    :return: tuple              - dx and dy index from an image and a corresponding error
    """
    arr_fit_c = arr_fit[cut_fit:-cut_fit, cut_fit:-cut_fit]
    arr_ref_c = arr_ref[cut_ref:-cut_ref, cut_ref:-cut_ref]
    cut_off = (cut_ref - cut_fit) * 2
    i, j, min_err = (0, 0, 0)
    for s_i in range(len(steps_)):
        step_ = steps_[s_i]
        if s_i == 0:
            max_i, max_j = (cut_off, cut_off)
            min_i, min_j = (0, 0)
        else:
            step_back = steps_[s_i - 1]
            max_i, max_j = (i + step_back, j + step_back)
            if max_i > cut_off:
                max_i = cut_off
            if max_j > cut_off:
                max_j = cut_off
            min_i, min_j = (i - step_back, j - step_back)
            if min_i < 0:
                min_i = 0
            if min_j < 0:
                min_j = 0
        err_ = img_match_err(arr_ref=arr_ref_c, arr_fit=arr_fit_c, iter_=step_, title=f" Matching {s_i + 1}",
                             max_i=max_i, min_i=min_i, max_j=max_j, min_j=min_j, cut_off=cut_off)
        err_ls = [err_i[2] for err_i in err_]

        min_err = min(err_ls)
        min_idx = err_ls.index(min_err)
        i, j = (err_[min_idx][0], err_[min_idx][1])


    dx = i - (cut_ref - cut_fit)
    dy = j - (cut_ref - cut_fit)

    return dx, dy, min_err


def match_adj(trans_i, trans_j):
    """
    Function to test adjust the translation parameters based on the translation results from both sides
    trans_i: image(t0) -> image(t1)
    trans_j: image(t1) -> image(t0)
    :param trans_i: tuple       -  x and y index from an image and a corresponding error of i-th translation
    :param trans_j: tuple       -  x and y index from an image and a corresponding error of j-th translation
    :param logs: MessageLog     - log class for printing the logs
    :return tuple:              - adjusted i-th translation
    :return tuple:              - adjusted j-th translation
    :return tuple:              - x and y residua
    """
    x_fit_adj = (trans_i[0] + trans_j[0]) / 2
    y_fit_adj = (trans_i[1] + trans_j[1]) / 2
    i_1 = trans_i[0] - x_fit_adj
    j_1 = trans_i[1] - y_fit_adj
    i_2 = trans_j[0] - x_fit_adj
    j_2 = trans_j[1] - y_fit_adj
    return (i_1, j_1, trans_i[2]), (i_2, j_2, trans_j[2]), (x_fit_adj, y_fit_adj)


def get_trans_param(arr_m, arr_s, steps_, cut_ref=355, cut_fit=115):
    """
    Function to get the translation parameters for two images (master and slave)
    :param arr_m: 2d np.array   - master image
    :param arr_s: 2d np.array   - slave image
    :param steps_: list/tuple   - list of steps for (intervals in a for loop)
    :param cut_ref: int         - cut size for one side for reference image
    :param cut_fit: int         - cut size for one side for image to fit
    :param logs: MessageLog     - log class for printing the logs
    :return: tuple              - adjusted translations and their residua
    """
    if steps_ is None:
        steps_ = [1]

    trans_s2m = img_match_trans(arr_ref=arr_m, arr_fit=arr_s, steps_=steps_, cut_ref=cut_ref,
                                cut_fit=cut_fit)
    trans_m2s = img_match_trans(arr_ref=arr_s, arr_fit=arr_m, steps_=steps_, cut_ref=cut_ref,
                                cut_fit=cut_fit)

    return match_adj(trans_i=trans_s2m, trans_j=trans_m2s)




def batch_temporal_upsampling(arr_m, arr_s, sev_m_sar_dif, sev_s_sar_dif, nd_val, 
                              batch_size_x, batch_size_y, overlap, steps_, 
                              cut_ref=355, cut_fit=115):
    """
    Function to perform temporal upsampling on full images in batches with overlap.
    For each batch: compute translation parameters using get_trans_param, 
    then apply temporal_upsampling, and finally merge all batches by averaging overlaps.
    
    :param arr_m: 2d np.array                   - full master image
    :param arr_s: 2d np.array                   - full slave image
    :param sev_m_sar_dif: datetime.timedelta    - time difference between reference and master
    :param sev_s_sar_dif: datetime.timedelta    - time difference between reference and slave
    :param nd_val: float                        - value for no data pixels
    :param batch_size_x: int                    - batch size in x direction
    :param batch_size_y: int                    - batch size in y direction
    :param overlap: int                         - overlap between batches
    :param steps_: list/tuple                   - steps for get_trans_param
    :param cut_ref: int                         - cut size for reference in get_trans_param
    :param cut_fit: int                         - cut size for fit in get_trans_param
    :return: 2d np.array                        - full upsampled image
    """
    H, W = arr_m.shape
    step_y = batch_size_y - overlap
    step_x = batch_size_x - overlap
    
    # Initialize result and count arrays
    result = np.zeros_like(arr_m, dtype=float)
    count = np.zeros_like(arr_m, dtype=int)
    
    y = 0
    while y < H:
        y_end = min(y + batch_size_y, H)
        x = 0
        while x < W:
            x_end = min(x + batch_size_x, W)
            
            # Extract batch
            tile_m = arr_m[y:y_end, x:x_end]
            tile_s = arr_s[y:y_end, x:x_end]
            
            # Get translation parameters
            trans = get_trans_param(tile_m, tile_s, steps_, cut_ref, cut_fit)
            m_dx, m_dy = trans[2]  # average translation
            
            # Perform temporal upsampling on the batch
            upsampled = temporal_upsampling(tile_m, tile_s, m_dx, m_dy, 
                                           sev_m_sar_dif, sev_s_sar_dif, nd_val)
            
            # Add to result and increment count
            result[y:y_end, x:x_end] += upsampled
            count[y:y_end, x:x_end] += 1
            
            x += step_x
        
        y += step_y
    
    # Average the overlaps
    result /= count
    
    return result


