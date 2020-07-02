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
from sklearn.metrics import ndcg_score
from utils import LoadSave

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)


def get_z_normalized_ts(ts=None):
    mean_val, std_val = np.mean(ts), np.std(ts)

    if std_val == 0:
        return ts
    else:
        return (ts - mean_val) / std_val


def load_data(path_name=None):
    """Loading *.pkl from path_name, path_name is like: .//data//mnist.pkl"""
    file_processor = LoadSave()
    return file_processor.load_data(path=path_name)


def jaccard_similarity_score(y_true=None, y_pred=None, k=None):
    """
    Computing the TOP-K jaccard similarity of two array: y_true, y_pred. 
    """
    if k == None:
        k = len(y_true)
    if y_true == None or y_pred == None or k > len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Invalid input parameters !")

    set_y_true, set_y_pred = set(y_true[:k]), set(y_pred[:k])

    intersection = set_y_true.intersection(set_y_pred)
    return len(intersection) / k


def plot_experiment_time_cost(experiment_res_list=None,
                              save_fig=True, dataset_name="heartbeat_mit"):
    """Plot the time cost of multi-experiment results."""
    if not isinstance(experiment_res_list, list):
        raise TypeError("Invalid experiment result type !")
    plt.close("all")

    # Plot 1: Average total searching time of each query ts on different size of dataset
    fig, ax = plt.subplots(figsize=(8, 5))
    for ind, experiment_res in enumerate(experiment_res_list):
        file_name_keys = list(experiment_res.keys())
        test_sample_size_list, mean_time_spend_list, std_time_spend_list = [], [], []
        for name in file_name_keys:
            test_sample_size = int(name.split("_")[-1])
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
                    markersize=5, linewidth=1.6, linestyle="-", color="k",
                    label="Optimized {}".format(ind))

        ax.set_xlabel("Searched Dataset set", fontsize=12)
        ax.set_ylabel("Time[s]", fontsize=12)
        ax.set_title("The time searched on a total dataset", fontsize=12)
        ax.tick_params(axis="both", labelsize=10, rotation=0)
        ax.set_xlim(min(test_sample_size_list), max(test_sample_size_list))
        ax.set_ylim(0, )
        ax.legend(fontsize=10)
        ax.grid(True)

    if save_fig:
        plt.savefig(".//plots//{}_experiment_time.png".format(dataset_name),
                    bbox_inches="tight", dpi=700)
    plt.close("all")


def plot_ndcg_performance(experiment_res_list=None, k=None,
                          save_fig=True, dataset_name="heartbeat_mit"):
    """Plot the NDCG scores of multi-experiment results."""
    if not isinstance(experiment_res_list, list):
        raise TypeError("Invalid experiment result type !")
    plt.close("all")
    baseline_experiment_res = experiment_res_list[0]

    k = [32, 64, 128, None]
    # Plot 1: NDCG Scores
    fig, ax_objs = plt.subplots(2, 2, figsize=(14, 10))
    ax_objs = ax_objs.ravel()

    for ind, experiment_res in enumerate(experiment_res_list[1:]):
        file_name_keys = list(experiment_res.keys())

        test_sample_size_list = []
        mean_ndcg_list, std_ndcg_list = [], []
        for name in file_name_keys:
            test_sample_size = int(name.split("_")[-1])
            test_sample_size_list.append(test_sample_size)

            baseline = baseline_experiment_res[name]
            optimized = experiment_res[name]
            ndcg_scores = []
            for ts_query in baseline.keys():
                query_baseline_res = sorted(baseline[ts_query]["top_n_searching_res"], key=lambda x: x[0])
                query_optimized_res = sorted(optimized[ts_query]["top_n_searching_res"], key=lambda x: x[0])

                # +0.0001 prevent ZeroDiv error
                query_baseline_res_array = [1 / (query_baseline_res[i][1] + 0.0001) for i in range(len(query_baseline_res))]
                query_optimized_res_array = [1 / (query_optimized_res[i][1] + 0.0001) for i in range(len(query_optimized_res))]

                # Score calculation: NDCG, AP, Jaccard Similarity
                tmp_score = []
                for top_k in k:
                    tmp_score.append(ndcg_score([query_baseline_res_array], [query_optimized_res_array], k=top_k))
                ndcg_scores.append(tmp_score)

            mean_ndcg_list.append(np.mean(ndcg_scores, axis=0))
            std_ndcg_list.append(np.std(ndcg_scores, axis=0))
        mean_ndcg_list = np.vstack(mean_ndcg_list)
        std_ndcg_list = np.vstack(std_ndcg_list)

        # x-axis: total-dataset-size, y-axis: NDCG Score
        for i, ax in enumerate(ax_objs):
            ax.plot(test_sample_size_list, mean_ndcg_list[:, i], marker="o",
                    markersize=5, linewidth=1.6, linestyle="-", color="k",
                    label="Optimized {}".format(ind))
            ax.fill_between(test_sample_size_list,
                            mean_ndcg_list[:, i] - std_ndcg_list[:, i],
                            mean_ndcg_list[:, i] + std_ndcg_list[:, i],
                            alpha=0.4, color="g")
    
            ax.set_xlabel("Dataset Size", fontsize=12)
            ax.set_ylabel("@NDCG", fontsize=12)
            ax.set_title("@NDCG(TOP-{}) Scores on the different dataset size".format(k[i]),
                         fontsize=12)
            ax.tick_params(axis="both", labelsize=10, rotation=0)
            ax.set_xlim(min(test_sample_size_list), max(test_sample_size_list))
            ax.legend(fontsize=10)
            ax.grid(True)
        plt.tight_layout()

    if save_fig:
        plt.savefig(".//plots//{}_experiment_ndcg.png".format(dataset_name),
                    bbox_inches="tight", dpi=700)
    plt.close("all")


def plot_jaccard_performance(experiment_res_list=None, k=None,
                             save_fig=True, dataset_name="heartbeat_mit"):
    """Plot the Jaccard scores of multi-experiment results."""
    if not isinstance(experiment_res_list, list):
        raise TypeError("Invalid experiment result type !")
    plt.close("all")
    baseline_experiment_res = experiment_res_list[0]

    k = [8, 16, 32, 64]
    # Plot 1: NDCG Scores
    fig, ax_objs = plt.subplots(2, 2, figsize=(14, 10))
    ax_objs = ax_objs.ravel()

    for ind, experiment_res in enumerate(experiment_res_list[1:]):
        file_name_keys = list(experiment_res.keys())

        test_sample_size_list = []
        mean_jaccard_list, std_jaccard_list = [], []
        for name in file_name_keys:
            test_sample_size = int(name.split("_")[-1])
            test_sample_size_list.append(test_sample_size)

            baseline = baseline_experiment_res[name]
            optimized = experiment_res[name]
            jaccard_scores = []
            for ts_query in baseline.keys():
                query_baseline_res = baseline[ts_query]["top_n_searching_res"]
                query_optimized_res = optimized[ts_query]["top_n_searching_res"]

                # +0.0001 prevent ZeroDiv error
                query_baseline_res_array = [query_baseline_res[i][0] for i in range(len(query_baseline_res))]
                query_optimized_res_array = [query_optimized_res[i][0] for i in range(len(query_optimized_res))]

                # Score calculation: NDCG, AP, Jaccard Similarity
                tmp_score = []
                for top_k in k:
                    tmp_score.append(jaccard_similarity_score(query_baseline_res_array, query_optimized_res_array, k=top_k))
                jaccard_scores.append(tmp_score)

            mean_jaccard_list.append(np.mean(jaccard_scores, axis=0))
            std_jaccard_list.append(np.std(jaccard_scores, axis=0))
        mean_jaccard_list = np.vstack(mean_jaccard_list)
        std_jaccard_list = np.vstack(std_jaccard_list)

        # x-axis: total-dataset-size, y-axis: NDCG Score
        for i, ax in enumerate(ax_objs):
            ax.plot(test_sample_size_list, mean_jaccard_list[:, i], marker="o",
                    markersize=5, linewidth=1.6, linestyle="-", color="k",
                    label="Optimized {}".format(ind))
            ax.fill_between(test_sample_size_list,
                            mean_jaccard_list[:, i] - std_jaccard_list[:, i],
                            mean_jaccard_list[:, i] + std_jaccard_list[:, i],
                            alpha=0.4, color="g")
    
            ax.set_xlabel("Dataset Size", fontsize=12)
            ax.set_ylabel("@Jaccard Similarity", fontsize=12)
            ax.set_title("@JACCARD(TOP-{}) Scores on the different dataset size".format(k[i]),
                         fontsize=12)
            ax.tick_params(axis="both", labelsize=10, rotation=0)
            ax.set_xlim(min(test_sample_size_list), max(test_sample_size_list))
            ax.legend(fontsize=10)
            ax.grid(True)
        plt.tight_layout()

    if save_fig:
        plt.savefig(".//plots//{}_experiment_jaccard.png".format(dataset_name),
                    bbox_inches="tight", dpi=700)
    plt.close("all")


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
    experiment_res_list = [load_data(path_name=".//data_tmp//" + dataset_name + "_baseline_searching_res.pkl"),
                            load_data(path_name=".//data_tmp//" + dataset_name + "_optimized_searching_res.pkl")]

    # plot_experiment_time_cost(experiment_res_list, dataset_name=dataset_name)
    # plot_jaccard_performance(experiment_res_list, dataset_name=dataset_name)

    # plot_top_n_similar_ts(dataset[-1], experiment_res, dataset_name=file_names[-1],
    #                       ts_query_ind=295, n=5)
