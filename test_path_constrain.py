#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:52:50 2020

@author: zhuoyin94
"""

import numpy as np
from tslearn.metrics import dtw as ts_dtw
from pyts.metrics import dtw as py_dtw, dtw_fast
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from utils import LoadSave
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)


def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


def sakoe_chiba_mask(sz1, sz2, radius=1):
    """
     The Sakoe-Chiba region is defined through a window_size parameter which
     determines the largest temporal shift allowed from the diagonal in the
     direction of the longest time series.
    """
    mask = np.full((sz1, sz2), -1)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in range(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in range(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


def _njit_itakura_mask(sz1, sz2, max_slope=2.):
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = np.empty((2, sz2))
    lower_bound[0] = min_slope * np.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * np.arange(sz2))
    lower_bound_ = np.empty(sz2)
    for i in range(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, sz2))
    upper_bound[0] = max_slope * np.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * np.arange(sz2))
    upper_bound_ = np.empty(sz2)
    for i in range(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    mask = np.full((sz1, sz2), -1)
    for i in range(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in range(sz1):
        if not np.any(np.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in range(sz2):
            if not np.any(np.isfinite(mask[:, j])):
                raise_warning = True
                break
    return mask


if __name__ == "__main__":
    heart_beat_dataset = load_data(
        path_name=".//data//heartbeat_mit_512.pkl")
    turnout_dataset = load_data(
        path_name=".//data//fault_turnout_current_512.pkl")

    use_dataset = 2
    seq_ind_0, seq_ind_1 = 210, 332

    if use_dataset == 0:
        seq_0, seq_1 = heart_beat_dataset[seq_ind_0], heart_beat_dataset[seq_ind_1]
    elif use_dataset == 1:
        seq_0, seq_1 = turnout_dataset[seq_ind_0][0], turnout_dataset[seq_ind_1][0]
    else:
        np.random.seed(2020)
        seq_0 = np.random.randn(140)
        seq_1 = np.random.randn(50)

    # Experiment 1: Different DTW caculation
    print("[INFO] DTW Calculation:")
    print("-- tslearn dtw: {:.5f}".format(ts_dtw(seq_0, seq_1)))
    print("-- pyts dtw: {:.5f}".format(py_dtw(seq_0, seq_1)))

    # Experiment 2: FastDTW calculation
    print("\n[INFO] FastDTW Calculation:")
    print("-- FastDTW results: {:.5f}".format(
        np.sqrt(fastdtw(seq_0, seq_1, radius=2, dist=lambda x, y: (x-y)**2)[0])))
    print("-- pyts FastDTW: {:.5f}".format(
        py_dtw(seq_0, seq_1, method="fast", options={"radius": 2})))

    # Experiment 3: Sakoe_Chiba calculation
    print("\n[INFO] Sakoe_Chiba Calculation:")
    print("-- tslearn Sakoe_Chiba dtw: {:.5f}".format(
        ts_dtw(seq_0, seq_1, sakoe_chiba_radius=5)))
    print("-- pyts Sakoe_Chiba dtw: {:.5f}".format(
        py_dtw(seq_0, seq_1, method="sakoechiba",  options={"window_size": 5})))

    # Experiment 4: itakura calculation
    print("\n[INFO] itakura Calculation:")
    print("-- tslearn itakura dtw: {:.5f}".format(
        ts_dtw(seq_0, seq_1, itakura_max_slope=6)))
    print("-- pyts itakura dtw: {:.5f}".format(
        py_dtw(seq_0, seq_1, method="itakura",  options={"max_slope": 6})))

    # # Plot sakoe_chiba warping path
    # radius = 1
    # mask = sakoe_chiba_mask(sz1, sz2, radius=radius)
    # plt.close("all")
    # f, ax = plt.subplots(figsize=(12, 10))
    # ax = sns.heatmap(mask, vmin=-1, vmax=1, ax=ax, linewidths=0.2)
    # ax.tick_params(axis="both", labelsize=12, rotation=0)
    # plt.tight_layout()

    # Plot sakoe_chiba warping path
    # max_slope = 4
    # mask = itakura_mask(sz1, sz2, max_slope=max_slope)
    # plt.close("all")
    # f, ax = plt.subplots(figsize=(12, 10))
    # ax = sns.heatmap(mask, vmin=-1, vmax=1, ax=ax, linewidths=0.2)
    # ax.tick_params(axis="both", labelsize=12, rotation=0)
    # plt.tight_layout()
