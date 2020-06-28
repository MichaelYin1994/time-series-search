#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:48:50 2020

@author: zhuoyin94
"""

"""
1. The time cost for searching the whole dataset.
2. Accuracy of the Top-K similarity searching results
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


def get_z_normalized_ts(ts=None):
    mean_val, std_val = np.mean(ts), np.std(ts)

    if std_val == 0:
        return ts
    else:
        return (ts - mean_val) / std_val


def get_jaccard_dist(set_x=None, set_y=None):
    set_x, set_y = set(set_x), set(set_y)
    union_set = set_x.union(set_y)
    intersection_set = set_x.intersection(set_y)

    return len(intersection_set)/len(union_set)


def plot_experiment_time_cost(experiment_res_list=None):
    """Plot the time cost of multi-experiment results."""
    if not isinstance(experiment_res_list, list):
        raise TypeError("Invalid experiment result type !")

    # Plot 1: Average total searching time of each ts
    fig, ax = plt.subplots(figsize=(8, 5))
    for ind, experiment_res in enumerate(experiment_res_list):

        file_name_keys = list(experiment_res.keys())
        test_sample_size_list, mean_time_spend_list, std_time_spend_list = [], [], []
        for name in file_name_keys:
            test_sample_size = int(name.split("_")[-1][:-4])
            test_sample_size_list.append(test_sample_size)

            mean_time_spend = np.mean([item["total_time_spend"] for item in experiment_res[name].values()])
            std_time_spned = np.std([item["total_time_spend"] for item in experiment_res[name].values()])
            mean_time_spend_list.append(mean_time_spend)
            std_time_spend_list.append(std_time_spned)

        # x-axis: total-dataset-size, y-axis: average searching time
        if ind == 0:
            ax.plot(test_sample_size_list, mean_time_spend_list, marker="o",
                    markersize=5, linewidth=1.6, linestyle="--", color="b",
                    label="baseline")
        else:
            ax.plot(test_sample_size_list, mean_time_spend_list, marker="o",
                    markersize=5, linewidth=1.6, linestyle="-", color="k")
        ax.fill_between(test_sample_size_list,
                        np.array(mean_time_spend_list) - np.array(std_time_spend_list),
                        np.array(mean_time_spend_list) + np.array(std_time_spend_list),
                        alpha=0.4, color="g")
        ax.tick_params(axis="both", labelsize=10, rotation=0)
        ax.set_xlim(0, max(test_sample_size_list))
        ax.set_ylim(0, )
        ax.legend(fontsize=10)
        ax.grid(True)
    return None


def plot_top_n_similar_ts(dataset=None, experiment_res=None,
                          dataset_name=None, ts_query_ind=None, n=5):
    """Plot top-n similar time series for each experiment_res."""
    top_n_ind = experiment_res[dataset_name][ts_query_ind]["top_n_searching_res"][1:]

    fig, ax_objs = plt.subplots(1, n+1, figsize=(17, 2),
                                sharex=True, sharey=True)
    ax_objs = ax_objs.ravel()

    ax = ax_objs[0]
    ts = get_z_normalized_ts(dataset[ts_query_ind])
    ax.plot(ts, linewidth=2, color="k", label="Query Time Series")

    for ind, ax in enumerate(ax_objs[1:]):
        ts = get_z_normalized_ts(dataset[int(top_n_ind[ind][0])])
        ax.plot(ts, linewidth=2, color="b", label="Query Time Series")

    for ax in ax_objs:
        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        ax.tick_params(axis="x", labelsize=9, rotation=0)
    fig.tight_layout(pad=0.1)


if __name__ == "__main__":
    PATH = ".//data//"
    dataset_name = "heartbeat_mit"
    file_names = os.listdir(PATH)

    file_names = [name for name in file_names if dataset_name in name]
    file_names = sorted(file_names, key=lambda s: int(s.split("_")[-1][:-4]))

    dataset = [load_data(PATH+name) for name in file_names]
    experiment_res_list = [load_data(path_name=".//data_tmp//" + dataset_name + "_searching_res.pkl"),
                           load_data(path_name=".//data_tmp//" + dataset_name + "_optimized_searching_res.pkl")]

    plot_experiment_time_cost(experiment_res_list)
    # plot_top_n_similar_ts(dataset[-1], experiment_res, dataset_name=file_names[-1],
    #                       ts_query_ind=295, n=5)
