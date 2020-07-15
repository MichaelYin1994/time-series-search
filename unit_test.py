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
from utils import LoadSave
from time import time
from tqdm import tqdm

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)

def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


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
        ax.set_ylim(0, )
        ax.tick_params(axis="both", labelsize=9, rotation=0)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=0.1)


def plot_compared_ts(dataset, ts_x_ind, ts_y_ind):
    fig, ax = plt.subplots(figsize=(8, 4))
    for ind in [ts_x_ind, ts_y_ind]:
        ts = dataset[ind]
        ax.plot(ts, linewidth=2, color="b", label="ind_{}".format(ind))
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        ax.tick_params(axis="both", labelsize=9, rotation=0)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=0.1)


def lb_kim_hierarchy(ts_query, ts_candidate, bsf):
    pass


def lb_keogh_cumulative(order, ts_query, ts_candidate, bsf):
    pass


def dtw_early_stop(ts_query, ts_candidate, bsf):
    pass


if __name__ == "__main__":
    dataset = load_data(path_name=".//data//heartbeat_ptbdb_512.pkl")


