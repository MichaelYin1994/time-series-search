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


def dist(x, y):
    return (x-y)*(x-y)


def lb_kim_hierarchy(ts_query, ts_candidate, bsf):
    """Reference can be seen in the source code of UCR DTW."""
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
    

def lb_keogh_cumulative(order, ts_query, ts_candidate, bsf):
    pass


def dtw_early_stop(ts_query, ts_candidate, bsf):
    pass


if __name__ == "__main__":
    # Preparing data
    N_TS_GENERATING = 100
    LEN_TS = 150

    dataset = []
    for i in range(N_TS_GENERATING):
        dataset.append(get_z_normalized_ts(np.random.rand(LEN_TS)))

    ts_x_ind, ts_y_ind = 10, 50
    ts_x, ts_y = dataset[ts_x_ind], dataset[ts_y_ind]

    # lb_kim_hierarchy
    lb_kim = lb_kim_hierarchy(ts_x, ts_y, 3.5)