import numpy as np


def add_x_start(arr, size, val):
    """
    Function to add array of values (val) at start of the x-axis
    :param arr: 2d np.array - initial array
    :param size: int        - x-size of the added array
    :param val: float       - value to fill
    :return: 2d np.array    - output array
    """
    sh_add = (size, np.shape(arr)[1])
    arr_add = np.empty(sh_add)
    arr_add[:] = val
    return np.append(arr_add, arr, axis=0)


def add_x_end(arr, size, val):
    """
    Function to add array of values (val) at end of the x-axis
    :param arr: 2d np.array - initial array
    :param size: int        - x-size of the added array
    :param val: float       - value to fill
    :return: 2d np.array    - output array
    """
    sh_add = (size, np.shape(arr)[1])
    arr_add = np.empty(sh_add)
    arr_add[:] = val
    return np.append(arr, arr_add, axis=0)


def add_y_start(arr, size, val):
    """
    Function to add array of values (val) at start of the y-axis
    :param arr: 2d np.array - initial array
    :param size: int        - y-size of the added array
    :param val: float       - value to fill
    :return: 2d np.array    - output array
    """
    sh_add = (np.shape(arr)[0], size)
    arr_add = np.empty(sh_add)
    arr_add[:] = val
    return np.append(arr_add, arr, axis=1)


def add_y_end(arr, size, val):
    """
    Function to add array of values (val) at end of the y-axis
    :param arr: 2d np.array - initial array
    :param size: int        - y-size of the added array
    :param val: float       - value to fill
    :return: 2d np.array    - output array
    """
    sh_add = (np.shape(arr)[0], size)
    arr_add = np.empty(sh_add)
    arr_add[:] = val
    return np.append(arr, arr_add, axis=1)


def cut2sh(arr_m, arr_s):
    """
    Function to cut by one pixel image bigger than the other one
    :param arr_m: 2d np.array       - master image
    :param arr_s: 2d np.array       - slave image
    :return arr_m_n: 2d np.array    - cut master image (if bigger than slave image)
    :return arr_s_n: 2d np.array    - cut slave image (if bigger than master image)
    """
    arr_m_n, arr_s_n = (np.copy(arr_m), np.copy(arr_s))
    sh_m = np.shape(arr_m_n)
    sh_s = np.shape(arr_s_n)
    if sh_m[0] < sh_s[0]:
        arr_s_n = arr_s_n[:-1, :]
    elif sh_m[0] > sh_s[0]:
        arr_m_n = arr_m_n[:-1, :]
    if sh_m[1] < sh_s[1]:
        arr_s_n = arr_s_n[:, :-1]
    elif sh_m[1] > sh_s[1]:
        arr_m_n = arr_m_n[:, :-1]
    return arr_m_n, arr_s_n