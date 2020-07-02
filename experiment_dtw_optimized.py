#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:23:05 2020

@author: zhuoyin94
"""

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw
from utils import LoadSave

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


def search_top_n_similar_ts(ts_query=None, data=None, n=10):
    """For the query ts, search the top-n similar ts in data object, return
       the searching result.
    """
    top_n_searching_res, time_spend = [], []
    for ind, ts in enumerate(data):
        start = time()
        dtw_dist = dtw(ts_query, ts, global_constraint="sakoe_chiba",
                       sakoe_chiba_radius=20)
        end = time()

        top_n_searching_res.append([ind, dtw_dist, end-start])
        time_spend.append(end-start)

    # Sorted the results
    top_n_searching_res = sorted(top_n_searching_res, key=lambda t: t[1])
    top_n_searching_res = top_n_searching_res[:n]

    searching_res = {}
    searching_res["top_n_searching_res"] = top_n_searching_res
    searching_res["mean_time_per_ts"] = np.mean(time_spend)
    searching_res["std_time_per_ts"] = np.std(time_spend)
    searching_res["total_searched_ts"] = len(data)
    searching_res["total_time_spend"] = np.sum(time_spend)
    return searching_res


if __name__ == "__main__":
    N_NEED_SEARCH = 256
    PATH = ".//data//"
    target_dataset_name = "heartbeat_mit"
    dataset_names = os.listdir(PATH)
    dataset_names = [name for name in dataset_names if target_dataset_name in name]
    dataset_names = sorted(dataset_names, key=lambda s: int(s.split("_")[-1][:-4]))

    dataset = [load_data(PATH+name) for name in dataset_names]
    dataset_names = [name[:-4] for name in dataset_names]
    experiment_total_res = {name: None for name in dataset_names}

    for data, name in zip(dataset, dataset_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data = [get_z_normalized_ts(ts) for ts in data]

        # STEP 1: Randomly sampled n ts from the raw dataset
        selected_ts_ind = sample_n_ts(data, n=N_NEED_SEARCH)

        # STEP 2: For each selected ts, search TOP_K_NEED_SEARCH ts in the raw dataset,
        #         return the top-k list results.
        search_res = {}
        for ts_ind in selected_ts_ind:
            ts_query = data[ts_ind]
            search_res_tmp = search_top_n_similar_ts(ts_query, data, n=len(data)-1)
            search_res[ts_ind] = search_res_tmp

        # STEP 3: Save the SEARCH_TOP_K results in experiment_res
        experiment_total_res[name] = search_res

    file_processor = LoadSave()
    new_file_name = ".//data_tmp//" + target_dataset_name + "_optimized_searching_res.pkl"
    file_processor.save_data(path=new_file_name,
                             data=experiment_total_res)
