#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:29:34 2019

@author: yinzhuo
"""


import time
import pickle
import warnings
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from numba import njit, prange

warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def timefn(fcn):
    """Decorator for efficency analysis. """
    @wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time


@njit
def dist(x, y):
    return (x-y)*(x-y)


@njit
def dtw_ucrdtw(ts_x,
               ts_y,
               cb_cum,
               window_size=5,
               bsf=np.inf):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------

    @Description:
    ----------
    Dynamic Time Warping(DTW) Distance with early stopping and the
    Sakoe-Chiba Band. The time complexity is O(n^2), but strictly less
    than O(n^2). The space complexity is O(window_size).

    @Parameters:
    ----------
    ts_x: {array-like}
        The time series x.
    ts_y: {array-like}
        The time series y.
    cb_cum: {array-like}
        Cummulative bound of lb_keogh array in the reverse form.
    window_size: {int-like}
        The window size of sakoe-chiba band.
    bsf: {float-like}
        The best-so-far query dtw distance.

    @Return:
    ----------
    dtw distance between ts_x and ts_y
    """
    l1, l2 = ts_x.shape[0], ts_y.shape[0]
    cum_sum = np.full((l1+1, l2+1), np.inf)
    cum_sum[0, 0] = 0
    bsf = bsf**2

    for i in prange(l1):
        row_min = np.inf
        for j in prange(l2):
            cum_sum[i+1, j+1] = dist(ts_x[i], ts_y[j])
            cum_sum[i+1, j+1] += min(cum_sum[i, j+1], cum_sum[i+1, j], cum_sum[i, j])

            # Calculating the row min for early stopping
            if cum_sum[i+1, j+1] < row_min:
                row_min = cum_sum[i+1, j+1]

        if (i+1 < l1) and ((row_min + cb_cum[i+1]) > bsf):
            return None
    return cum_sum[-1, -1]**0.5


class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data
