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
from utils import LoadSave, dtw_ucrdtw
from time import time
from tqdm import tqdm
import heapq as hq
from numba import njit, prange

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)


def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


def preprocessing_ts(ts=None, envelope_radius=30):
    """Z-Normalized the ts, at the same time, then compute the lb_keogh for the ts."""
    ts = np.array(ts)
    mean_val, std_val = ts.mean(), ts.std()
    if std_val == 0:
        std_val = 0.00001

    # Step 1: statistical figures calculation
    ts_norm = (ts - mean_val) / std_val
    min_ind, max_ind = ts_norm.argmin(), ts_norm.argmax()
    min_val, max_val = ts_norm.min(), ts_norm.max()

    # Step 2: Envelope of time series
    lb_keogh_down, lb_keogh_up = lb_envelope(ts_norm, radius=envelope_radius)

    # Step 3: Argsort the ts
    ts_ind_ordered = np.argsort(np.abs(ts_norm))[::-1]
    return np.array([min_ind, max_ind, min_val, max_val]), ts_norm, lb_keogh_down.reshape(-1, ), lb_keogh_up.reshape(-1, ), ts_ind_ordered


@njit
def dist(x, y):
    return (x-y)*(x-y)


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
def lb_keogh_reverse_cumulative(cb, cb_cum):
    cb_cum[len(cb_cum)-1] = cb[len(cb)-1]
    for i in prange(len(cb_cum)-2, 0-1, -1):
        cb_cum[i] = cb_cum[i+1] + cb[i]
    return cb_cum


def search_top_n_similar_ts(ts_query_compact=None,
                            data_compact=None,
                            n=10,
                            verbose=False):
    """For the query ts, search the top-n similar ts in data object, return
       the searching result.
    """
    min_heap, time_spend = [], 0
    lb_kim_puring_count = 0
    lb_keogh_puring_count = 0
    lb_keogh_ec_puring_count = 0
    es_puring_count = 0

    start = time()
    ts_query = ts_query_compact[1]
    ts_query_ind_ordered = ts_query_compact[4]
    lb_keogh_query, ub_keogh_query = ts_query_compact[2], ts_query_compact[3]

    for ind, ts_candidate_compact in enumerate(data_compact):
        ts_candidate = ts_candidate_compact[1]
        ts_candidate_ind_ordered = ts_candidate_compact[4]
        lb_keogh_candidate, ub_keogh_candidate = ts_candidate_compact[2], ts_candidate_compact[3]

        # Initializing minimum heap(n + 1 for excluding itself)
        if len(min_heap) < n + 1:
            dtw_dist = -dtw_ucrdtw(ts_query, ts_candidate, np.zeros(len(ts_query)))
            # dtw_dist = -dtw(ts_query, ts_candidate_compact[1])
            hq.heappush(min_heap, [dtw_dist, ind])
            continue

        bsf = min_heap[0][0]
        cb, cb_ec, cb_cum = np.empty(len(ts_query)), np.empty(len(ts_query)), np.zeros(len(ts_query))
        # STEP 1: lb_kim_hierarchy puring
        # -------------------        
        lb_kim = -lb_kim_hierarchy(ts_query, ts_candidate, -bsf)
        if lb_kim < bsf:
            lb_kim_puring_count += 1
            continue

        # Enhance the lb_kim using the pre-computed maximum and minimum value
        if int(ts_query_compact[0][0]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3] \
            and int(ts_candidate_compact[0][0]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3]:
            pass
        else:
            lb_kim += -dist(ts_query_compact[0][2], ts_candidate_compact[0][2])

        if int(ts_query_compact[0][1]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3] \
            and int(ts_candidate_compact[0][1]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3]:
            pass
        else:
            lb_kim += -dist(ts_query_compact[0][3], ts_candidate_compact[0][3])
        if lb_kim < bsf:
            lb_kim_puring_count += 1
            continue

        # STEP 2: LB_Keogh puring(Including exchange)
        # -------------------
        lb_keogh_original, cb = lb_keogh_cumulative(ts_query_ind_ordered,
                                                    lb_keogh_query,
                                                    ub_keogh_query,
                                                    cb,
                                                    ts_candidate,
                                                    -bsf)
        lb_keogh_original = -lb_keogh_original
        if lb_keogh_original < bsf:
            lb_keogh_puring_count += 1
            continue

        lb_keogh_ec, cb_ec = lb_keogh_cumulative(ts_candidate_ind_ordered, 
                                                 lb_keogh_candidate,
                                                 ub_keogh_candidate,
                                                 cb_ec,
                                                 ts_query,
                                                 -bsf)
        lb_keogh_ec = -lb_keogh_ec
        if lb_keogh_ec < bsf:
            lb_keogh_ec_puring_count += 1
            continue

        # STEP 3: Computing the DTW distance
        # -------------------
        if lb_keogh_original > lb_keogh_ec:
            cb_cum = lb_keogh_reverse_cumulative(cb_ec, cb_cum)
        else:
            cb_cum = lb_keogh_reverse_cumulative(cb, cb_cum)

        dtw_dist = dtw_ucrdtw(ts_query,
                              ts_candidate,
                              cb_cum,
                              bsf=-bsf)
        if dtw_dist is None:
            es_puring_count += 1
            continue

        dtw_dist = -dtw_dist
        if dtw_dist < bsf:
            continue
        else:
            hq.heapreplace(min_heap, [dtw_dist, ind])

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
    min_heap = [[item[1], -item[0]] for item in min_heap]
    min_heap = sorted(min_heap, key=lambda x: x[1])[1:]

    searching_res = {}
    searching_res["top_n_searching_res"] = min_heap
    searching_res["mean_time_per_ts"] = time_spend / len(data)
    searching_res["std_time_per_ts"] = np.nan
    searching_res["total_searched_ts"] = len(data)
    searching_res["total_computed_ts_precent"] = (1 - lb_kim_puring_count/len(data) - lb_keogh_puring_count/len(data) - lb_keogh_ec_puring_count/len(data)) * 100
    searching_res["total_time_spend"] = time_spend

    searching_res["LB_Kim"] = lb_kim_puring_count/len(data) * 100
    searching_res["LB_Keogh"] = lb_keogh_puring_count/len(data) * 100
    searching_res["LB_Keogh_EC"] = lb_keogh_ec_puring_count/len(data) * 100
    searching_res["ES_Puring"] = es_puring_count/len(data) * 100
    searching_res["DTW_count"] = (1 - lb_kim_puring_count/len(data) - lb_keogh_puring_count/len(data) - lb_keogh_ec_puring_count/len(data)) * 100
    return searching_res


def load_benchmark(dataset_name=None):
    benchmark_dataset = load_data(path_name=".//data_tmp//" + dataset_name + "_baseline_searching_res.pkl")
    return benchmark_dataset


if __name__ == "__main__":
    N_NEED_SEARCH = 256
    TOP_N_SEARCH = 3
    PATH = ".//data//"
    TARGET_DATASET_NAME = "heartbeat_ptbdb"

    # Loading all the dataset
    # ---------------------------
    dataset_names = [name for name in os.listdir(PATH) if TARGET_DATASET_NAME in name]
    dataset_names = sorted(dataset_names, key=lambda s: int(s.split("_")[-1][:-4]))[-1:]

    dataset = [load_data(PATH+name) for name in dataset_names]
    dataset_names = [name[:-4] for name in dataset_names]
    experiment_total_res = {name: None for name in dataset_names}

    benchmark_dataset = load_benchmark(TARGET_DATASET_NAME)
    benchmark_dataset = {name: benchmark_dataset[name] for name in dataset_names}

    # Searching experiment start
    # ---------------------------
    for data, name in zip(dataset, dataset_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data_compact = []
        for i in range(len(data)):
            data_compact.append(preprocessing_ts(data[i], envelope_radius=60))

        # STEP 1: Randomly sampled n ts from the raw dataset
        selected_ts_ind = np.random.choice(list(benchmark_dataset[name].keys()),
                                           N_NEED_SEARCH, replace=False)

        # STEP 2: For each selected ts, search TOP_K_NEED_SEARCH ts in the raw dataset,
        #         return the top-k list results.
        search_res = {}
        time_accelerate, error_rate = [], []
        time_spend, computed_precent = [], []

        precent_lb_kim_puring = []
        precent_early_stop_puring = []
        precent_lb_keogh_puring, precent_lb_keogh_ec_puring = [], []
        precent_dtw = []
        for ts_ind in tqdm(selected_ts_ind):
            ts_query_compact = data_compact[ts_ind]
            search_res[ts_ind] = search_top_n_similar_ts(ts_query_compact, data_compact,
                                                         n=TOP_N_SEARCH, verbose=False)

            benchmark = benchmark_dataset[name][ts_ind]
            # print("[INFO] Time accelerate: {:.5f}".format(
            #     benchmark["total_time_spend"] / search_res_tmp["total_time_spend"]))

            # Accuracy checking
            benchmark_res_tmp = [item[0] for item in benchmark["top_n_searching_res"][:TOP_N_SEARCH]]
            search_res_tmp = [item[0] for item in search_res[ts_ind]["top_n_searching_res"]]
            checking_res = (np.array(search_res_tmp) == np.array(benchmark_res_tmp)).sum()

            error = 0
            if checking_res != TOP_N_SEARCH:
                error = 1

            time_spend.append(search_res[ts_ind]["total_time_spend"])
            time_accelerate.append(benchmark["total_time_spend"] / search_res[ts_ind]["total_time_spend"])
            computed_precent.append(search_res[ts_ind]["total_computed_ts_precent"])
            error_rate.append(error)

            precent_lb_kim_puring.append(search_res[ts_ind]["LB_Kim"])
            precent_early_stop_puring.append(search_res[ts_ind]["ES_Puring"])
            precent_lb_keogh_puring.append(search_res[ts_ind]["LB_Keogh"])
            precent_lb_keogh_ec_puring.append(search_res[ts_ind]["LB_Keogh_EC"])
            precent_dtw.append(search_res[ts_ind]["DTW_count"])

        # STEP 3: Save the SEARCH_TOP_N results in experiment_res
        experiment_total_res[name] = search_res

        # Print all information
        print("\n[INFO] Time spend each query: {:.5f} +- {:.5f}".format(
            np.mean(time_spend), np.std(time_spend)))
        print("[INFO] Accelerated compared with basleine each query: {:.5f} +- {:.5f}".format(
            np.mean(time_accelerate), np.std(time_accelerate)))
        print("[INFO] Computed precent each query: {:.5f}% +- {:.5f}%".format(
            np.mean(computed_precent), np.std(computed_precent)))
        print("[INFO] Error rate: {:.5f}".format(sum(error_rate) / N_NEED_SEARCH))
        print("[INFO] Puring precent(LB_Kim): {:.5f}%".format(
            np.mean(precent_lb_kim_puring)))
        print("[INFO] Puring precent(LB_Keogh): {:.5f}%".format(
            np.mean(precent_lb_keogh_puring)))
        print("[INFO] Puring precent(LB_Keogh_EC): {:.5f}%".format(
            np.mean(precent_lb_keogh_ec_puring)))
        print("[INFO] Puring precent(Early Stopping): {:.5f}%".format(
            np.mean(precent_early_stop_puring)))
        print("[INFO] DTW count: {:.5f}%".format(
            np.mean(precent_dtw)))

    # file_processor = LoadSave()
    # new_file_name = ".//data_tmp//" + target_dataset_name + "_ucrdtw_searching_res.pkl"
    # file_processor.save_data(path=new_file_name,
    #                           data=experiment_total_res)