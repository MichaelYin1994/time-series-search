#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:01:45 2020

@author: zhuoyin94
"""

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw, lb_keogh, lb_envelope
import _ucrdtw
from utils import LoadSave, dtw_early_stop
from time import time
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
    """Normlaizing the ts, at the same time, compute the LB_Kim."""
    ts = np.array(ts)
    mean_val, std_val = ts.mean(), ts.std()
    if std_val == 0:
        std_val = 0.00001

    ts_norm = (ts - mean_val) / std_val
    min_val_ind, max_val_ind = ts_norm.argmin(), ts_norm.argmax()
    if min_val_ind == 0 or min_val_ind == (len((ts_norm)) - 1):
        min_val = np.nan
    else:
        min_val = ts_norm[min_val_ind]

    if max_val_ind == 0 or max_val_ind == (len((ts_norm)) - 1):
        max_val = np.nan
    else:
        max_val = ts_norm[max_val_ind]

    return np.array([min_val, max_val, ts_norm[0], ts_norm[-1]]), ts_norm


def search_top_n_similar_ts(ts_query_compact=None, data=None, n=10, verbose=False):
    """For the query ts, search the top-n similar ts in data object, return
       the searching result.
    """
    min_n_heap, time_spend = [], 0
    lb_kim_puring_count = 0
    lb_keogh_puring_count, lb_keogh_ec_puring_count = 0, 0
    es_puring_count = 0

    start = time()
    lb_kim_query, ts_query = ts_query_compact[0], ts_query_compact[1]
    for ind, ts_candidate_compact in enumerate(data):

        # Initializing minimum heap(n + 1 for excluding itself)
        if len(min_n_heap) < n + 1:
            dtw_dist = -dtw_early_stop(ts_query, ts_candidate_compact[1], np.inf)
            # dtw_dist = -dtw(ts_query, ts_candidate_compact[1])
            hq.heappush(min_n_heap, [dtw_dist, ind])
            continue

        # Step 1: LB_Kim puring(np.nansum for avoiding duplicate sum)
        lb_kim_candidate = ts_candidate_compact[0]
        lb_kim = -np.sqrt(np.nansum(np.square(lb_kim_query - lb_kim_candidate)))

        if lb_kim < min_n_heap[0][0]:
            lb_kim_puring_count += 1
            continue

        # Step 2: LB_Keogh puring(Including exchange)
        ts_candidate = ts_candidate_compact[1]

        lb_keogh_original = -lb_keogh(ts_query, ts_candidate, radius=35)
        if lb_keogh_original < min_n_heap[0][0]:
            lb_keogh_puring_count += 1
            continue

        lb_keogh_ec = -lb_keogh(ts_candidate, ts_query, radius=35)
        if lb_keogh_ec < min_n_heap[0][0]:
            lb_keogh_ec_puring_count += 1
            continue

        # Step 3: Computing the DTW distance
        best_so_far = -min_n_heap[0][0]
        # dtw_dist = -dtw(ts_query, ts_candidate[1])
        dtw_dist = -dtw_early_stop(ts_query, ts_candidate, best_so_far)
        if np.isnan(dtw_dist):
            es_puring_count += 1
            continue
        if dtw_dist < min_n_heap[0][0]:
            continue
        else:
            hq.heapreplace(min_n_heap, [dtw_dist, ind])

        # top_n_searching_res.append([ind, dtw_dist, end-start])
    end = time()
    time_spend = end - start

    if verbose:
        print("[INFO] Time spend: {:.5f}".format(time_spend))
        print("[INFO] LB_Kim: {:.5f}%, LB_Keogh: {:.5f}%, LB_Keogh_ec: {:.5f}%, es_puring: {:.5f}%, DTW: {:.5f}%".format(
            lb_kim_puring_count/len(data) * 100,
            lb_keogh_puring_count/len(data) * 100,
            lb_keogh_ec_puring_count/len(data) * 100,
            es_puring_count/len(data) * 100,
            (1 - lb_kim_puring_count/len(data) - lb_keogh_puring_count/len(data) - lb_keogh_ec_puring_count/len(data)) * 100))

    # Sorted the results, exclude the first results(QUery itself)
    min_n_heap = [[item[1], -item[0]] for item in min_n_heap]
    min_n_heap = sorted(min_n_heap, key=lambda x: x[1])[1:]

    searching_res = {}
    searching_res["top_n_searching_res"] = min_n_heap
    searching_res["mean_time_per_ts"] = time_spend / len(data)
    searching_res["std_time_per_ts"] = np.nan
    searching_res["total_searched_ts"] = len(data)
    searching_res["total_time_spend"] = time_spend
    return searching_res


def load_benchmark(dataset_name=None):
    benchmark_dataset = load_data(path_name=".//data_tmp//" + dataset_name + "_baseline_searching_res.pkl")
    return benchmark_dataset


if __name__ == "__main__":
    N_NEED_SEARCH = 16
    TOP_N_SEARCH = 16
    PATH = ".//data//"
    target_dataset_name = "heartbeat_mit"
    dataset_names = os.listdir(PATH)
    dataset_names = [name for name in dataset_names if target_dataset_name in name]
    dataset_names = sorted(dataset_names, key=lambda s: int(s.split("_")[-1][:-4]))[-1:]

    # Loading all the dataset
    dataset = [load_data(PATH+name) for name in dataset_names]
    dataset_names = [name[:-4] for name in dataset_names]
    experiment_total_res = {name: None for name in dataset_names}

    benchmark_dataset = load_benchmark(target_dataset_name)
    benchmark_dataset = {name: benchmark_dataset[name] for name in dataset_names}

    for data, name in zip(dataset, dataset_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data_new = []
        for i in range(len(data)):
            data_new.append(get_z_normalized_ts(data[i]))

        # STEP 1: Randomly sampled n ts from the raw dataset
        selected_ts_ind = sample_n_ts(data_new, n=N_NEED_SEARCH)

        # STEP 2: For each selected ts, search TOP_K_NEED_SEARCH ts in the raw dataset,
        #         return the top-k list results.
        search_res = {}
        for ts_ind in selected_ts_ind:
            print("ts_ind: {}".format(ts_ind))
            ts_query_compact = data_new[ts_ind]
            search_res_tmp = search_top_n_similar_ts(ts_query_compact, data_new,
                                                      n=TOP_N_SEARCH, verbose=True)
            search_res[ts_ind] = search_res_tmp

            # benchmark = benchmark_dataset[name][ts_ind]
            # print("[INFO] Time accelerate: {:.5f}".format(
            #     benchmark["total_time_spend"] / search_res_tmp["total_time_spend"]))

            # # Accuracy checking
            # benchmark_res_tmp = [item[0] for item in benchmark["top_n_searching_res"][:TOP_N_SEARCH]]
            # search_res_tmp = [item[0] for item in search_res_tmp["top_n_searching_res"]]
            # checking_res = (np.array(search_res_tmp) == np.array(benchmark_res_tmp)).sum()
            # if checking_res != TOP_N_SEARCH:
            #     print("[ERROR] MIS-MATCHING: {:.5f}%".format((TOP_N_SEARCH - checking_res)/TOP_N_SEARCH * 100))
            #     # raise ValueError("Searhing result mismatch !")
            # print("\n")
        # STEP 3: Save the SEARCH_TOP_N results in experiment_res
        experiment_total_res[name] = search_res

    # file_processor = LoadSave()
    # new_file_name = ".//data_tmp//" + target_dataset_name + "_ucrdtw_searching_res.pkl"
    # file_processor.save_data(path=new_file_name,
    #                          data=experiment_total_res)