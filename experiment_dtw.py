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
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from tslearn.metrics import dtw
from utils import LoadSave, lb_kim_hierarchy
from tqdm import tqdm
import heapq as hq

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
    mean_val, std_val = np.mean(ts, axis=0), np.std(ts, axis=0)
    return (ts - mean_val) / std_val


def get_unnorm_ts(ts=None):
    return ts


def search_top_n_similar_ts(ts_query=None,
                            data=None,
                            n=10,
                            use_lb_kim=False):
    """For the query ts, search the top-n similar ts in data object, return
       the searching result.
    """
    start = time()
    min_heap, time_spend = [], 0

    for ind, ts_candidate in enumerate(data):
        # Initializing minimum heap(n + 1 for excluding itself)
        if len(min_heap) < n + 1:
            dtw_dist = -dtw(ts_query, ts_candidate)
            hq.heappush(min_heap, [dtw_dist, ind])
            continue

        # STEP 1: lb_kim_hierarchy puring
        # -------------------
        bsf = min_heap[0][0]
        if use_lb_kim:
            lb_kim = -np.sqrt(lb_kim_hierarchy(ts_query, ts_candidate, bsf**2))
            if lb_kim < bsf:
                continue

        # STEP 2: DTW calculation
        # -------------------
        dtw_dist = -dtw(ts_query, ts_candidate)
        if dtw_dist < bsf:
            continue
        else:
            hq.heapreplace(min_heap, [dtw_dist, ind])
    end = time()
    time_spend = end - start

    # Saving results
    top_n_searching_res = sorted(min_heap, key=lambda t: -t[0])
    top_n_searching_res = [[-t[0], t[1]] for t in top_n_searching_res][1:]

    searching_res = {}
    searching_res["top_n_searching_res"] = top_n_searching_res
    searching_res["total_searched_ts"] = len(data)
    searching_res["total_time_spend"] = time_spend
    return searching_res


if __name__ == "__main__":
    N_INSTANCE_NEED_TO_SEARCH = 512
    KEEP_TOP_N = 16
    DATA_PATH = ".//data//"

    # human_activity_recognition, heartbeat_mit, heartbeat_ptbdb
    TARGET_DATASET_NAME = "heartbeat_ptbdb"
    USE_LB_KIM = True
    CHECK_1NN_ACC = True
    NORM_TS = True
    SAVE_EXPERIMENT_RESULTS = True
    ###########################################################################

    # Loading all dataset with key word: TARGET_DATASET_NAME
    dataset_names = os.listdir(DATA_PATH)
    dataset_names = [name for name in dataset_names if TARGET_DATASET_NAME in name]
    dataset_names = sorted(dataset_names, key=lambda s: int(s.split("_")[-1][:-4]))

    dataset = [load_data(DATA_PATH+name) for name in dataset_names]
    raw_dataset = [item[0] for item in dataset]
    raw_label = [item[1] for item in dataset]

    dataset_names = [name[:-4] for name in dataset_names]
    experiment_total_res = {name: None for name in dataset_names}

    # Search TOP-N for each selected sequence
    # -------------------
    for data, data_label, name in zip(raw_dataset, raw_label, dataset_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data_norm = []
        for ts in data:
            if NORM_TS:
                data_norm.append(get_z_normalized_ts(ts))
            else:
                data_norm.append(get_unnorm_ts(ts))

        # STEP 1: Randomly sampled n ts from the raw dataset
        selected_ts_ind = sample_n_ts(data_norm, n=N_INSTANCE_NEED_TO_SEARCH)

        # STEP 2: For each selected ts, search TOP_K_NEED_SEARCH ts in the raw dataset,
        #         return the top-k list results.
        search_res, acc_list = {}, []
        print("\n[INFO] DATASET NAME: {}".format(name))
        for ts_ind in tqdm(selected_ts_ind):
            ts_query = data_norm[ts_ind]
            search_res[ts_ind] = search_top_n_similar_ts(ts_query,
                                                         data_norm,
                                                         n=KEEP_TOP_N,
                                                         use_lb_kim=USE_LB_KIM)

            if CHECK_1NN_ACC:
                one_nn_label = data_label[search_res[ts_ind]["top_n_searching_res"][0][1]]
                one_nn_dist = search_res[ts_ind]["top_n_searching_res"][0][0]

                true_label = data_label[ts_ind]
                acc_list.append(one_nn_label == true_label)

        # STEP 3: Save the SEARCH_TOP_K results in experiment_res
        experiment_total_res[name] = search_res
        if CHECK_1NN_ACC:
            print("\n[INFO] Mean 1-NN accuracy: {:.5f}.".format(
                np.mean(acc_list)))

    if SAVE_EXPERIMENT_RESULTS:
        file_processor = LoadSave()
        new_file_path = ".//data_tmp//{}_baseline_top_{}.pkl".format(
            TARGET_DATASET_NAME, KEEP_TOP_N)

        print("\n")
        file_processor.save_data(path=new_file_path,
                                 data=experiment_total_res)
