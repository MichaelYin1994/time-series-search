#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:58:40 2020

@author: zhuoyin94
"""

import matplotlib.pyplot as plt
import numpy as np
from pyts.metrics import sakoe_chiba_band
from pyts.metrics.dtw import _check_sakoe_chiba_params
from tslearn.metrics import sakoe_chiba_mask


def plot_sakoe_chiba_pyts(n_timestamps_1, n_timestamps_2, window_size=0.5, ax=None):
    """Plot the Sakoe-Chiba band."""
    region = sakoe_chiba_band(n_timestamps_1, n_timestamps_2, window_size)
    scale, horizontal_shift, vertical_shift = \
        _check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2, window_size)
    mask = np.zeros((n_timestamps_2, n_timestamps_1))
    for i, (j, k) in enumerate(region.T):
        mask[j:k, i] = 1.

    plt.imshow(mask, origin='lower', cmap='Wistia', vmin=0, vmax=1)

    sz = max(n_timestamps_1, n_timestamps_2)
    x = np.arange(-1, sz + 1)
    lower_bound = scale * (x - horizontal_shift) - vertical_shift
    upper_bound = scale * (x + horizontal_shift) + vertical_shift
    plt.plot(x, lower_bound, 'b', lw=2)
    plt.plot(x, upper_bound, 'g', lw=2)
    diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
    plt.plot(x, diag, 'black', lw=1)

    for i in range(n_timestamps_1):
        for j in range(n_timestamps_2):
            plt.plot(i, j, 'o', color='k', ms=3)

    ax.set_xticks(np.arange(-0.5, n_timestamps_1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_timestamps_2, 1), minor=True)
    plt.grid(color='b', which='minor', linestyle='--', linewidth=1)
    plt.xticks(np.arange(0, n_timestamps_1, 1))
    plt.yticks(np.arange(0, n_timestamps_2, 1))
    plt.xlim((-0.5, n_timestamps_1 - 0.5))
    plt.ylim((-0.5, n_timestamps_2 - 0.5))


def plot_sakoe_chiba_tslearn(n_timestamps_1, n_timestamps_2, window_size=0.5, ax=None):
    """Plot the Sakoe-Chiba band."""
    vertical_shift = window_size
    mask = sakoe_chiba_mask(n_timestamps_1, n_timestamps_2, window_size)
    mask[mask == 0.] = 1.
    mask[np.isinf(mask)] = 0.
    mask = mask.T

    plt.imshow(mask, origin='lower', cmap='Wistia', vmin=0, vmax=1)

    sz = max(n_timestamps_1, n_timestamps_2)
    x = np.arange(-1, sz + 1)
    lower_bound = x - vertical_shift - abs(n_timestamps_1 - n_timestamps_2)
    upper_bound = x + vertical_shift
    plt.plot(x, lower_bound, 'b', lw=2)
    plt.plot(x, upper_bound, 'g', lw=2)
    diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
    plt.plot(x, diag, 'black', lw=1)

    for i in range(n_timestamps_1):
        for j in range(n_timestamps_2):
            plt.plot(i, j, 'o', color='k', ms=3)

    ax.set_xticks(np.arange(-0.5, n_timestamps_1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_timestamps_2, 1), minor=True)
    plt.grid(color='b', which='minor', linestyle='--', linewidth=1)
    plt.xticks(np.arange(0, n_timestamps_1, 1))
    plt.yticks(np.arange(0, n_timestamps_2, 1))
    plt.xlim((-0.5, n_timestamps_1 - 0.5))
    plt.ylim((-0.5, n_timestamps_2 - 0.5))


if __name__ == "__main__":
    n_timestamps_1, n_timestamps_2 = 10, 7

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(2, 2, 1)
    plot_sakoe_chiba_pyts(n_timestamps_1, n_timestamps_2, window_size=0, ax=ax)
    plt.title('pyts, window-size = 0', fontsize=18)

    ax = plt.subplot(2, 2, 2)
    plot_sakoe_chiba_pyts(n_timestamps_1, n_timestamps_2, window_size=2, ax=ax)
    plt.title('pyts, window-size = 2', fontsize=18)

    ax = plt.subplot(2, 2, 3)
    plot_sakoe_chiba_tslearn(n_timestamps_1, n_timestamps_2, window_size=0, ax=ax)
    plt.title('tslearn, window-size = 0', fontsize=18)

    ax = plt.subplot(2, 2, 4)
    plot_sakoe_chiba_tslearn(n_timestamps_1, n_timestamps_2, window_size=2, ax=ax)
    plt.title('tslearn, window-size = 2', fontsize=18)

    # plt.suptitle('Sakoe-Chiba band', y=1.02, fontsize=24)
    # plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
