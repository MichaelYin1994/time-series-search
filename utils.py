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
from numba import njit

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
def _local_squared_dist(x, y):
    dist = 0.
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist


@njit
def _dtw_early_stop(s1, s2, best_so_far):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.
    es_flag = 0

    for i in range(l1):
        row_min = np.inf
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                         cum_sum[i + 1, j],
                                         cum_sum[i, j])

            # Calculating the row min for early stopping
            if cum_sum[i + 1, j + 1] < row_min:
                row_min = cum_sum[i + 1, j + 1]

        if row_min > best_so_far:
            es_flag = 1
        if es_flag:
            break

    if es_flag:
        return None
    return cum_sum[1:, 1:]


def dtw_early_stop(s1, s2, best_so_far):
    s1 = s1.reshape((-1, 1))
    s2 = s2.reshape((-1, 1))

    cum_sum_mat = _dtw_early_stop(s1, s2, best_so_far**2)
    if cum_sum_mat is None:
        return np.nan
    return np.sqrt(cum_sum_mat[-1, -1])




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
