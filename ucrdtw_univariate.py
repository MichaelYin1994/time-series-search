#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:01:38 2020

@author: zhuoyin94
"""

import os
import pickle
from time import time
from tqdm import tqdm
import heapq as hq
import numpy as np
from numba import njit, prange
from tslearn.metrics import lb_envelope


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


def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


def preprocessing_ts(ts=None,
                     envelope_radius=30,
                     is_norm_ts=True):
    """Z-Normalized the ts, at the same time, compute the lb_keogh for the ts."""
    # Z-norm the ts(Original paper: 1.2.1 Time Series Subsequences must be Normalized)
    if is_norm_ts:
        mean_val, std_val = np.mean(ts, axis=0), np.std(ts, axis=0)
        std_val = std_val + 0.0000001
        ts_norm = (ts - mean_val) / std_val
    else:
        ts_norm = ts

    # Basic Statistical
    min_ind, max_ind = ts_norm.argmin(axis=0), ts_norm.argmax(axis=0)
    min_val, max_val = ts_norm.min(axis=0), ts_norm.max(axis=0)

    # Envelope of time series(Original paper: 4.1.2 Lower Bounding)
    keogh_lb, keogh_ub = lb_envelope(ts_norm, radius=envelope_radius)

    # Argsort the ts(Original paper: 4.2.2 Reordering Early Abandoning)
    ts_ind_ordered = np.argsort(np.abs(ts_norm))[::-1]

    return np.array([min_ind, max_ind, min_val, max_val]), ts_norm, keogh_lb.reshape(-1, ), keogh_ub.reshape(-1, ), ts_ind_ordered


@njit
def dist(x, y):
    return (x-y)*(x-y)


@njit
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
    # TODO: O(n) space complexity DTW with Sakoe-Chiba Band
    l1, l2 = ts_x.shape[0], ts_y.shape[0]
    cum_sum = np.full((l1+1, l2+1), np.inf)
    cum_sum[0, 0] = 0

    for i in prange(l1):
        row_min = np.inf
        for j in prange(l2):
            cum_sum[i+1, j+1] = dist(ts_x[i], ts_y[j])
            cum_sum[i+1, j+1] += min(cum_sum[i, j+1], cum_sum[i+1, j], cum_sum[i, j])

            # Calculating the row min for early stopping
            if cum_sum[i+1, j+1] < row_min:
                row_min = cum_sum[i+1, j+1]

        if (i+1 < l1) and ((row_min + cb_cum[i+1]) > bsf):
            return None
    return cum_sum[-1, -1]


@njit
def lb_keogh_cumulative(ts_query_ind_order,
                        ts_query_lb,
                        ts_query_ub,
                        ts_query_cb,
                        ts_candidate,
                        bsf):
    """Reference can be seen in the source code of UCR_DTW(lb_keogh_cumulative)."""
    lb_keogh_dist = 0
    for i in prange(len(ts_candidate)):
        d = 0
        if ts_candidate[ts_query_ind_order[i]] > ts_query_ub[ts_query_ind_order[i]]:
            d = dist(ts_candidate[ts_query_ind_order[i]], ts_query_ub[ts_query_ind_order[i]])
        elif ts_candidate[ts_query_ind_order[i]] < ts_query_lb[ts_query_ind_order[i]]:
            d = dist(ts_candidate[ts_query_ind_order[i]], ts_query_lb[ts_query_ind_order[i]])

        lb_keogh_dist += d
        ts_query_cb[ts_query_ind_order[i]] = d
        if lb_keogh_dist > bsf:
            return lb_keogh_dist, ts_query_cb
    return lb_keogh_dist, ts_query_cb


def lb_kim_hierarchy(ts_query, ts_candidate, bsf):
    """Reference can be seen in the source code of UCR_DTW(lb_kim_hierarchy)."""
    # 1 point at front and end
    lb_kim = 0
    lb_kim += (dist(ts_query[0], ts_candidate[0]) + dist(ts_query[-1], ts_candidate[-1]))
    if lb_kim > bsf:
        return lb_kim

    # 2 points at front
    path_cost = min(dist(ts_query[0], ts_candidate[1]),
                    dist(ts_query[1], ts_candidate[0]),
                    dist(ts_query[1], ts_candidate[1]))
    lb_kim += path_cost
    if lb_kim > bsf:
        return lb_kim

    # 2 points at end
    path_cost = min(dist(ts_query[-2], ts_candidate[-2]),
                    dist(ts_query[-1], ts_candidate[-2]),
                    dist(ts_query[-2], ts_candidate[-1]))
    lb_kim += path_cost
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
    path_cost = min(dist(ts_query[2], ts_candidate[0]),
                    dist(ts_query[2], ts_candidate[1]),
                    dist(ts_query[2], ts_candidate[2]),
                    dist(ts_query[1], ts_candidate[2]),
                    dist(ts_query[0], ts_candidate[2]))
    lb_kim += path_cost
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
    path_cost = min(dist(ts_query[-3], ts_candidate[-3]),
                    dist(ts_query[-3], ts_candidate[-2]),
                    dist(ts_query[-3], ts_candidate[-1]),
                    dist(ts_query[-2], ts_candidate[-3]),
                    dist(ts_query[-1], ts_candidate[-3]))
    lb_kim += path_cost
    return lb_kim


@njit
def lb_keogh_reverse_cumulative(cb, cb_cum):
    cb_cum[len(cb_cum)-1] = cb[len(cb)-1]
    for i in prange(len(cb_cum)-2, 0-1, -1):
        cb_cum[i] = cb_cum[i+1] + cb[i]
    return cb_cum


def search_top_n_similar_ts(ts_query_compact=None,
                            ts_candidate_data_compact=None,
                            n=10,
                            is_use_lb_kim=True,
                            is_use_lb_keogh=True,
                            is_use_lb_keogh_ec=True,
                            is_use_early_stop=True):
    """For the query ts, search the top-n similar ts in data object, return
       the searching result.
    """
    min_heap = []
    lb_kim_puring_count = 0
    lb_keogh_puring_count = 0
    lb_keogh_ec_puring_count = 0
    es_puring_count = 0
    candidate_size = len(ts_candidate_data_compact)

    start = time()
    ts_query = ts_query_compact[1]
    ts_query_ind_ordered = ts_query_compact[4]
    lb_keogh_query, ub_keogh_query = ts_query_compact[2], ts_query_compact[3]

    for ind, ts_candidate_compact in enumerate(ts_candidate_data_compact):
        ts_candidate = ts_candidate_compact[1]
        ts_candidate_ind_ordered = ts_candidate_compact[4]
        lb_keogh_candidate, ub_keogh_candidate = ts_candidate_compact[2], ts_candidate_compact[3]

        # Initializing minimum heap(n + 1 for excluding itself)
        if len(min_heap) < n + 1:
            dtw_dist = -dtw_ucrdtw(ts_query, ts_candidate, np.zeros(len(ts_query)))
            hq.heappush(min_heap, [dtw_dist, ind])
            continue

        bsf = min_heap[0][0]
        cb, cb_ec, cb_cum = np.zeros(len(ts_query)), np.zeros(len(ts_query)), np.zeros(len(ts_query))

        # STEP 1: lb_kim_hierarchy puring
        # -------------------
        if is_use_lb_kim:
            lb_kim = -lb_kim_hierarchy(ts_query, ts_candidate, -bsf)
            if lb_kim < bsf:
                lb_kim_puring_count += 1
                continue

        # # Enhance the lb_kim using the pre-computed maximum and minimum value
        # if int(ts_query_compact[0][0]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3] \
        #     and int(ts_candidate_compact[0][0]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3]:
        #     pass
        # else:
        #     lb_kim += -dist(ts_query_compact[0][2], ts_candidate_compact[0][2])

        # if int(ts_query_compact[0][1]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3] \
        #     and int(ts_candidate_compact[0][1]) in [0, 1, 2, len(ts_query)-1, len(ts_query)-2, len(ts_query)-3]:
        #     pass
        # else:
        #     lb_kim += -dist(ts_query_compact[0][3], ts_candidate_compact[0][3])
        # if lb_kim < bsf:
        #     lb_kim_puring_count += 1
        #     continue

        # STEP 2: LB_Keogh puring(Including exchange)
        # -------------------
        if is_use_lb_keogh:
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

        if is_use_lb_keogh_ec:
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
        if is_use_lb_keogh and is_use_lb_keogh_ec:
            if lb_keogh_original > lb_keogh_ec:
                cb_cum = lb_keogh_reverse_cumulative(cb_ec, cb_cum)
            else:
                cb_cum = lb_keogh_reverse_cumulative(cb, cb_cum)

        if is_use_early_stop:
            dtw_dist = dtw_ucrdtw(ts_query,
                                  ts_candidate,
                                  cb_cum,
                                  bsf=-bsf)
        else:
            dtw_dist = dtw_ucrdtw(ts_query,
                                  ts_candidate,
                                  cb_cum,
                                  bsf=np.inf)
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

    # Sorted the results, exclude the first results(QUery itself)
    min_heap = [[item[1], -item[0]] for item in min_heap]
    min_heap = sorted(min_heap, key=lambda x: x[1])[1:]

    searching_res = {}
    searching_res["top_n_searching_res"] = min_heap
    searching_res["mean_time_per_ts"] = time_spend / candidate_size
    searching_res["std_time_per_ts"] = np.nan
    searching_res["total_searched_ts"] = candidate_size
    searching_res["total_computed_ts_precent"] = (1 - lb_kim_puring_count/candidate_size - lb_keogh_puring_count/candidate_size - lb_keogh_ec_puring_count/candidate_size) * 100
    searching_res["total_time_spend"] = time_spend

    searching_res["LB_Kim"] = lb_kim_puring_count/candidate_size * 100
    searching_res["LB_Keogh"] = lb_keogh_puring_count/candidate_size * 100
    searching_res["LB_Keogh_EC"] = lb_keogh_ec_puring_count/candidate_size * 100
    searching_res["ES_Puring"] = es_puring_count/candidate_size * 100
    searching_res["DTW_count"] = (1 - lb_kim_puring_count/candidate_size - lb_keogh_puring_count/candidate_size - lb_keogh_ec_puring_count/candidate_size) * 100
    return searching_res


if __name__ == "__main__":
    # heartbeat_mit, heartbeat_ptbdb
    DATASET_PATH = ".//data//"
    BENCHMARK_DATASET_PATH = ".//data_tmp//"
    TARGET_DATASET_NAME = "heartbeat_mit"
    BENCHMARK_DATASET_NAME = "heartbeat_mit_baseline_top_16"
    CHECK_1NN_ACC = True
    SAVE_EXPERIMENT_RESULTS = False

    N_INSTANCE_NEED_TO_SEARCH = 512
    KEEP_TOP_N = 3
    NORM_TS = True
    ENVELOPE_RADIUS = 30
    IS_USE_LB_KIM = False
    IS_USE_LB_KEOGH = True
    IS_USE_LB_KEOGH_EC = True
    IS_USE_EARLY_STOP = True # If one of IS_USE_LB_KEOGH and IS_USE_LB_KEOGH_EC is True, then IS_USE_EARLY_STOP is False

    ###########################################################################
    # Loading all the dataset
    # ---------------------------
    dataset_names = [name for name in os.listdir(DATASET_PATH) if TARGET_DATASET_NAME in name]
    dataset_names = sorted(dataset_names, key=lambda s: int(s.split("_")[-1][:-4]))[-1:]

    dataset = [load_data(DATASET_PATH+name) for name in dataset_names]
    dataset_names = [name[:-4] for name in dataset_names]
    raw_dataset = [item[0] for item in dataset]
    raw_label = [item[1] for item in dataset]
    experiment_total_res = {name: None for name in dataset_names}

    benchmark_dataset = load_data(path_name=BENCHMARK_DATASET_PATH + BENCHMARK_DATASET_NAME + ".pkl")
    benchmark_dataset = {name: benchmark_dataset[name] for name in dataset_names}

    ###########################################################################
    # Searching experiment start
    # ---------------------------
    for data, data_label, name in zip(raw_dataset, raw_label, dataset_names):
        # STEP 0: preprocessing ts(Normalized, Filtering outlier)
        data_compact = []
        for i in range(len(data)):
            data_compact.append(preprocessing_ts(data[i].reshape((-1, )),
                                                 envelope_radius=ENVELOPE_RADIUS))

        # STEP 1: Randomly sampled N_INSTANCE_NEED_TO_SEARCH ts from the benchmark_dataset keys
        selected_ts_ind = np.random.choice(list(benchmark_dataset[name].keys()),
                                           N_INSTANCE_NEED_TO_SEARCH, replace=False)

        # STEP 2: For each selected ts, search TOP_K_NEED_SEARCH ts in the raw dataset,
        #         return the top-k list results.
        search_res = {}
        time_accelerate, error_rate = [], []
        time_spend, computed_precent = [], []
        acc_list = []

        precent_lb_kim_puring = []
        precent_early_stop_puring = []
        precent_lb_keogh_puring, precent_lb_keogh_ec_puring = [], []
        precent_dtw = []
        for ts_ind in tqdm(selected_ts_ind):
            ts_query_compact = data_compact[ts_ind]
            search_res[ts_ind] = search_top_n_similar_ts(ts_query_compact,
                                                         data_compact,
                                                         n=KEEP_TOP_N,
                                                         is_use_lb_kim=IS_USE_LB_KIM,
                                                         is_use_lb_keogh=IS_USE_LB_KEOGH,
                                                         is_use_lb_keogh_ec=IS_USE_LB_KEOGH_EC,
                                                         is_use_early_stop=IS_USE_EARLY_STOP)

            benchmark = benchmark_dataset[name][ts_ind]
            # Accuracy checking
            benchmark_res_tmp = [item[1] for item in benchmark["top_n_searching_res"][:KEEP_TOP_N]]
            search_res_tmp = [item[0] for item in search_res[ts_ind]["top_n_searching_res"]]
            checking_res = (np.array(search_res_tmp) == np.array(benchmark_res_tmp)).sum()

            error = 0
            if checking_res != KEEP_TOP_N:
                error = 1

            time_spend.append(search_res[ts_ind]["total_time_spend"])
            # time_accelerate.append(benchmark["total_time_spend"] / search_res[ts_ind]["total_time_spend"])
            computed_precent.append(search_res[ts_ind]["total_computed_ts_precent"])
            error_rate.append(error)

            precent_lb_kim_puring.append(search_res[ts_ind]["LB_Kim"])
            precent_early_stop_puring.append(search_res[ts_ind]["ES_Puring"])
            precent_lb_keogh_puring.append(search_res[ts_ind]["LB_Keogh"])
            precent_lb_keogh_ec_puring.append(search_res[ts_ind]["LB_Keogh_EC"])
            precent_dtw.append(search_res[ts_ind]["DTW_count"])

            if CHECK_1NN_ACC:
                one_nn_label = data_label[search_res[ts_ind]["top_n_searching_res"][0][0]]
                true_label = data_label[ts_ind]
                acc_list.append(one_nn_label == true_label)

        # STEP 3: Save the SEARCH_TOP_N results in experiment_res
        experiment_total_res[name] = search_res

        # Print all information
        print("\n---------------------------------")
        print("DATASET NAME: {}, N_INSTANCE_NEED_TO_SEARCH: {}, KEEP_TOP_N: {}".format(
            name, N_INSTANCE_NEED_TO_SEARCH, KEEP_TOP_N))
        print("[INFO] Time spend per query: {:.5f} +- {:.5f}".format(
            np.mean(time_spend), np.std(time_spend)))
        # print("[INFO] Accelerated compared with basleine per query: {:.5f} +- {:.5f}".format(
        #     np.mean(time_accelerate), np.std(time_accelerate)))
        print("[INFO] DTW computed precent per query: {:.5f}% +- {:.5f}%".format(
            np.mean(computed_precent), np.std(computed_precent)))
        print("[INFO] Mis-ranking rate: {:.5f}%".format(sum(error_rate) / N_INSTANCE_NEED_TO_SEARCH * 100))
        print("[INFO] Classification accuracy: {:.5f}%".format(np.mean(acc_list) * 100))

        print("[INFO] Puring precent(LB_Kim): {:.5f}%".format(
            np.mean(precent_lb_kim_puring)))
        print("[INFO] Puring precent(LB_Keogh): {:.5f}%".format(
            np.mean(precent_lb_keogh_puring)))
        print("[INFO] Puring precent(LB_Keogh_EC): {:.5f}%".format(
            np.mean(precent_lb_keogh_ec_puring)))
        print("[INFO] Puring precent(Early Stopping): {:.5f}%".format(
            np.mean(precent_early_stop_puring)))
        print("[INFO] DTW Computed precent: {:.5f}%\n".format(
            np.mean(precent_dtw)))
