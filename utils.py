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
    return np.sum((x-y)*(x-y))


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
    Sakoe-Chiba band. The time complexity is O(n^2), but strictly less
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


def lb_kim_hierarchy(ts_query, ts_candidate, bsf):
    """Reference can be seen in the source code of UCR DTW."""
    # 1 point at front and end
    lb_kim = 0
    lb_kim += (dist(ts_query[0], ts_candidate[0]) + dist(ts_query[-1], ts_candidate[-1]))
    if lb_kim > bsf:
        return lb_kim

    # 2 points at front
    path_dist = min(dist(ts_query[0], ts_candidate[1]),
                    dist(ts_query[1], ts_candidate[0]),
                    dist(ts_query[1], ts_candidate[1]))
    lb_kim += path_dist
    if lb_kim > bsf:
        return lb_kim

    # 2 points at end
    path_dist = min(dist(ts_query[-2], ts_candidate[-2]),
                    dist(ts_query[-1], ts_candidate[-2]),
                    dist(ts_query[-2], ts_candidate[-1]))
    lb_kim += path_dist
    if lb_kim > bsf:
        return lb_kim

    # 3 pints at front:
    #
    #      0      1       2       3
    # 0  np.inf  np.inf  np.inf  np.inf
    # 1  np.inf   o       o       x
    # 2  np.inf   o       o       x
    # 3  np.inf   x       x       x
    #
    # Find the minimum distance among all (x)
    path_dist = min(dist(ts_query[2], ts_candidate[0]),
                    dist(ts_query[2], ts_candidate[1]),
                    dist(ts_query[2], ts_candidate[2]),
                    dist(ts_query[1], ts_candidate[2]),
                    dist(ts_query[0], ts_candidate[2]))
    lb_kim += path_dist
    if lb_kim > bsf:
        return lb_kim

    # 3 pints at end:
    #
    #     -3    -2    -1
    # -3   x     x     x
    # -2   x     o     o
    # -1   x     o     o
    #
    # Find the minimum distance among all (x)
    path_dist = min(dist(ts_query[-3], ts_candidate[-3]),
                    dist(ts_query[-3], ts_candidate[-2]),
                    dist(ts_query[-3], ts_candidate[-1]),
                    dist(ts_query[-2], ts_candidate[-3]),
                    dist(ts_query[-1], ts_candidate[-3]))
    lb_kim += path_dist
    return lb_kim


@njit
def lb_keogh_reverse_cumulative(cb, cb_cum):
    cb_cum[len(cb_cum)-1] = cb[len(cb)-1]
    for i in prange(len(cb_cum)-2, 0-1, -1):
        cb_cum[i] = cb_cum[i+1] + cb[i]
    return cb_cum


@njit
def lb_keogh_cumulative(ts_query_index_order,
                        ts_query_lb,
                        ts_query_ub,
                        ts_query_cb,
                        ts_candidate,
                        bsf):
    lb_keogh_, bsf = 0, bsf**2
    for i in prange(len(ts_candidate)):
        d = 0
        if ts_candidate[ts_query_index_order[i]] > ts_query_ub[ts_query_index_order[i]]:
            d = dist(ts_candidate[ts_query_index_order[i]], ts_query_ub[ts_query_index_order[i]])
        elif ts_candidate[ts_query_index_order[i]] < ts_query_lb[ts_query_index_order[i]]:
            d = dist(ts_candidate[ts_query_index_order[i]], ts_query_lb[ts_query_index_order[i]])

        lb_keogh_ += d
        ts_query_cb[ts_query_index_order[i]] = d
        if lb_keogh_ > bsf:
            return lb_keogh_**0.5, ts_query_cb
    return lb_keogh_**0.5, ts_query_cb


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
