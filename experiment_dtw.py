#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:22:58 2020

@author: zhuoyin94
"""

"""Randomly selected 100 samples from the raw dataset, find top-20 similar
time series from the raw dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import LoadSave

from tslearn.metrics import dtw, lb_keogh
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)

def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


def sample_n_ts(data=None, n=50):
    """Randomly sample n ts from data, return index of the data."""
    ind = np.arange(len(data))
    sampled_ind = np.random.choice(ind, n, replace=False)
    return sampled_ind


def get_z_normalized_ts(ts=None):
    mean_val, std_val = np.mean(ts), np.std(ts)

    if std_val == 0:
        return ts
    else:
        return (ts - mean_val) / std_val


def search_n_similar():
    pass


def lb_kim():
    pass


if __name__ == "__main__":
    N_TESTS = 10
    SEARCH_TOP_K = 50
    PATH = ".//data//"
    dataset_name = "heartbeat_mit"
    file_names = os.listdir(PATH)

    file_names = [name for name in file_names if dataset_name in name]
    file_names = sorted(file_names, key=lambda s: int(s.split("_")[-1][:-4]))

    dataset = [load_data(PATH+name) for name in file_names]
    experiment_res = {name: None for name in file_names}

    for data, name in zip(dataset, file_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data = [get_z_normalized_ts(ts) for ts in data]

        # STEP 1: Randomly sampled n ts from the raw dataset
        selected_ts_ind = sample_n_ts(data, n=N_TESTS)

        # Step 2: For each selected ts, search SEARCH_TOP_K ts in the raw dataset,
        #         return the top-k index.
        search_res = {}
        for ts_ind in selected_ts_ind:
            ts_query = data[ts_ind]
            search_res_tmp = search_n_similar(ts_query, data, n=SEARCH_TOP_K)
            search_res[ts_ind] = search_res_tmp

        # Step 3: Save the SEARCH_TOP_K results in experiment_res
        experiment_res[name] = search_res[ts_ind]
