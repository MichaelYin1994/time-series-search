#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:31:31 2020

@author: zhuoyin94
"""
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw, lb_keogh, lb_envelope
import _ucrdtw
from time import time
from tqdm import tqdm
from numba import njit, prange

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)


def get_z_normalized_ts(ts=None):
    mean_val, std_val = np.mean(ts), np.std(ts)

    if std_val == 0:
        return ts
    else:
        return (ts - mean_val) / std_val


def plot_random_n_ts(dataset=None, n=3):
    sampled_ind = np.random.choice(np.arange(len(dataset)), n, replace=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for ind in sampled_ind:
        ts = dataset[ind]
        ax.plot(ts, linewidth=2, color="b", label="ind_{}".format(ind))
        ax.set_xlim(0, )
        # ax.set_ylim(0, )
        ax.tick_params(axis="both", labelsize=9, rotation=0)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=0.1)


def plot_compared_ts(dataset, ts_x_ind, ts_y_ind):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["b", "r", "g", "k"]

    for color_ind, ts_ind in enumerate([ts_x_ind, ts_y_ind]):
        ts = dataset[ts_ind]
        ax.plot(ts, linewidth=2, color=colors[color_ind], label="ind_{}".format(ts_ind))
        ax.set_xlim(0, )
        # ax.set_ylim(0, )
        ax.tick_params(axis="both", labelsize=9, rotation=0)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=0.1)


@njit
def dist(x, y):
    return (x-y)*(x-y)


def lb_kim_hierarchy(ts_query,
                     ts_candidate,
                     bsf):
    """Reference can be seen in the source code of UCR DTW function lb_kim_hierarchy."""
    # 1 point at front and back
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
    # Finf the minimum distance among all (x)
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
    # Finf the minimum distance among all (x)
    path_dist = min(dist(ts_query[-3], ts_candidate[-3]),
                    dist(ts_query[-3], ts_candidate[-2]),
                    dist(ts_query[-3], ts_candidate[-1]),
                    dist(ts_query[-2], ts_candidate[-3]),
                    dist(ts_query[-1], ts_candidate[-3]))
    lb_kim += path_dist
    return lb_kim


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


@njit
def lb_keogh_reverse_cumulative(cb, cb_cum):
    cb_cum[len(cb_cum)-1] = cb[len(cb)-1]
    for i in prange(len(cb_cum)-2, 0-1, -1):
        cb_cum[i] = cb_cum[i+1] + cb[i]
    return cb_cum


if __name__ == "__main__":
    # Preparing data
    N_TS_GENERATING = 100
    LEN_TS = 150

    dataset = []
    for i in range(N_TS_GENERATING):
        dataset.append(get_z_normalized_ts(np.random.rand(LEN_TS)))

    ts_query_ind, ts_candidate_ind = 22, 50
    ts_query, ts_candidate = dataset[ts_query_ind], dataset[ts_candidate_ind]
    ts_query_ind_ordered, ts_candidate_ind_ordered = np.argsort(np.abs(ts_query))[::-1], np.argsort(np.abs(ts_candidate))[::-1]

    # lb_kim_hierarchy
    lb_kim = lb_kim_hierarchy(ts_query, ts_candidate, 3.5)

    # lb_keogh_cumulative
    cb, cb_ec = np.empty(len(ts_query)), np.empty(len(ts_query))
    lb_keogh_query, ub_keogh_query = lb_envelope(ts_query, radius=30)
    lb_keogh_query, ub_keogh_query = lb_keogh_query.reshape(-1, ), ub_keogh_query.reshape(-1, )

    lb_keogh_candidate, ub_keogh_candidate = lb_envelope(ts_candidate, radius=30)
    lb_keogh_candidate, ub_keogh_candidate = lb_keogh_candidate.reshape(-1, ), ub_keogh_candidate.reshape(-1, )

    lb_keogh_original, cb = lb_keogh_cumulative(ts_query_ind_ordered,
                                                lb_keogh_query,
                                                ub_keogh_query,
                                                cb,
                                                ts_candidate,
                                                np.inf)

    lb_keogh_ec, cb_ec = lb_keogh_cumulative(ts_candidate_ind_ordered, 
                                             lb_keogh_candidate,
                                             ub_keogh_candidate,
                                             cb_ec,
                                             ts_query,
                                             np.inf)

    cb_cum = np.zeros((len(ts_query, )))
    cb_cum = lb_keogh_reverse_cumulative(cb, cb_cum)
