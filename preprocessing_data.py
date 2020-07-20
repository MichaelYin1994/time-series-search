#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:50:21 2020

@author: zhuoyin94
"""

import numpy as np
import pandas as pd
from utils import LoadSave
np.random.seed(2099)


def preprocessing_turnout(n_data_list=None):
    file_processor = LoadSave()
    signal_data_list = []

    # Experiment fault data
    def liststr_to_listnumeric(list_str):
        return list(map(float, list_str.split(",")))
    signal_fault_data = pd.read_csv("..//demo_dataset//turnout//fault_data.csv",
                                    nrows=None).query("error_code != 0").reset_index(drop=True)
    signal_fault_data["Phase_A"] = signal_fault_data["Phase_A"].apply(liststr_to_listnumeric)
    signal_fault_data["Phase_B"] = signal_fault_data["Phase_B"].apply(liststr_to_listnumeric)
    signal_fault_data["Phase_C"] = signal_fault_data["Phase_C"].apply(liststr_to_listnumeric)

    for i in range(len(signal_fault_data)):
        signal = [signal_fault_data["Phase_A"].iloc[i],
                  signal_fault_data["Phase_B"].iloc[i],
                  signal_fault_data["Phase_C"].iloc[i]]
        signal_data_list.append(signal)

    # Operation fault data
    signal_data = file_processor.load_data(
        path="..//demo_dataset//turnout//chengdu5_raw_table.pkl")
    signal_anomaly_scores = file_processor.load_data(
        path="..//demo_dataset//turnout//chengdu5_anomaly_scores.pkl")
    signal_data = pd.merge(signal_data, signal_anomaly_scores,
                           on=["device_id", "record_id"], how="left")
    signal_data = signal_data.sort_values(
        by="if_score", ascending=False).reset_index(drop=True)

    for i in range(len(signal_data)):
        signal = [signal_data["phase_a"].iloc[i],
                  signal_data["phase_b"].iloc[i],
                  signal_data["phase_c"].iloc[i]]
        signal_data_list.append(signal)

    # Save the proprocessed data
    for ind in range(len(n_data_list)):
        if n_data_list[ind] is None:
            n_data_list[ind] = len(signal_data_list)

    file_name = [".//data//fault_turnout_current_{}.pkl".format(i)
                 for i in n_data_list]
    for ind, item in enumerate(n_data_list):
        tmp_signal_data = signal_data_list[:item]
        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name, data=tmp_signal_data)


def preprocessing_mnist(n_data_list=None):
    img_data = pd.read_csv("..//demo_dataset//mnist//train.csv",
                           nrows=None)
    img_cols = [name for name in img_data.columns if "pixel" in name]
    img_data = img_data[img_cols].values / 255
    img_data_list = img_data.tolist()

    # Save the proprocessed data
    file_name = [".//data//mnist_{}.pkl".format(i)
                 for i in n_data_list]
    file_processor = LoadSave()
    for ind, item in enumerate(n_data_list):
        tmp_img_data = img_data_list[:item]
        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name, data=tmp_img_data)


def preprocessing_fashion_mnist():
    img_data = pd.read_csv("..//demo_dataset//fashion_mnist//fashionmnist//fashion_mnist_train.csv",
                           nrows=None)
    img_cols = [name for name in img_data.columns if "pixel" in name]
    img_data = img_data[img_cols].values / 255
    img_data_list = img_data.tolist()

    # Save the proprocessed data
    file_name = [".//data//fashion_mnist_{}.pkl".format(i)
                 for i in n_data_list]
    file_processor = LoadSave()
    for ind, item in enumerate(n_data_list):
        tmp_img_data = img_data_list[:item]
        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name, data=tmp_img_data)


def preprocessing_heartbeat_mit(n_data_list=None):
    heartbeat_data = pd.read_csv("..//demo_dataset//heartbeat//heartbeat//mitbih_train.csv",
                                 nrows=None, header=None)
    heartbeat_data = heartbeat_data.sample(frac=1).reset_index(drop=True)

    data_cols = [i for i in range(heartbeat_data.shape[1] - 1)]
    data_labels = int(heartbeat_data.shape[1]) - 1

    heartbeat_label_list = heartbeat_data[data_labels].astype(int).tolist()
    heartbeat_data = heartbeat_data[data_cols].values
    heartbeat_data_list = heartbeat_data.tolist()

    # Save the proprocessed data
    for ind in range(len(n_data_list)):
        if n_data_list[ind] is None:
            n_data_list[ind] = len(heartbeat_data_list)

    file_name = [".//data//heartbeat_mit_{}.pkl".format(i)
                 for i in n_data_list]
    file_processor = LoadSave()
    for ind, item in enumerate(n_data_list):
        tmp_data = heartbeat_data_list[:item]
        tmp_data_label = heartbeat_label_list[:item]

        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name,
                                 data=[tmp_data, tmp_data_label])


def preprocessing_heartbeat_ptbdb(n_data_list=None):
    tmp_data_0 = pd.read_csv("..//demo_dataset//heartbeat//heartbeat//ptbdb_normal.csv",
                                 nrows=None, header=None)
    tmp_data_1 = pd.read_csv("..//demo_dataset//heartbeat//heartbeat//ptbdb_abnormal.csv",
                                 nrows=None, header=None)
    heartbeat_data = pd.concat([tmp_data_0, tmp_data_1], axis=0)
    heartbeat_data = heartbeat_data.sample(frac=1).reset_index(drop=True)

    data_cols = [i for i in range(heartbeat_data.shape[1] - 1)]
    data_labels = int(heartbeat_data.shape[1]) - 1

    heartbeat_label_list = heartbeat_data[data_labels].astype(int).tolist()
    heartbeat_data = heartbeat_data[data_cols].values
    heartbeat_data_list = heartbeat_data.tolist()

    # Save the proprocessed data
    for ind in range(len(n_data_list)):
        if n_data_list[ind] is None:
            n_data_list[ind] = len(heartbeat_data_list)

    file_name = [".//data//heartbeat_ptbdb_{}.pkl".format(i)
                  for i in n_data_list]
    file_processor = LoadSave()
    for ind, item in enumerate(n_data_list):
        tmp_data = heartbeat_data_list[:item]
        tmp_data_label = heartbeat_label_list[:item]

        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name,
                                 data=[tmp_data, tmp_data_label])


def preprocessing_HAR(n_data_list=None):
    file_processor = LoadSave()
    har_dataset, har_dataset_label = file_processor.load_data(
        path="..//demo_dataset//human_activity_recognition//human_activity_recognition.pkl")
    har_dataset = har_dataset.reshape((-1, 8, 62))

    for ind in range(len(n_data_list)):
        if n_data_list[ind] is None:
            n_data_list[ind] = len(har_dataset)

    file_name = [".//data//human_activity_recognition_{}.pkl".format(i)
                 for i in n_data_list]
    file_processor = LoadSave()
    for ind, item in enumerate(n_data_list):
        tmp_data = har_dataset[:item]
        tmp_data_label = har_dataset_label[:item]

        tmp_file_name = file_name[ind]
        file_processor.save_data(path=tmp_file_name,
                                 data=[tmp_data, tmp_data_label])


if __name__ == "__main__":
    n_data_list = [512, 8192, 16384, None]
    preprocessing_heartbeat_mit(n_data_list.copy())

    n_data_list = [512, 8192, None]
    preprocessing_heartbeat_ptbdb(n_data_list.copy())

    n_data_list = [512, None]
    preprocessing_HAR(n_data_list.copy())