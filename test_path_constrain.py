#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:52:50 2020

@author: zhuoyin94
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)


def sakoe_chiba_mask(sz1, sz2, radius=1):
    mask = np.full((sz1, sz2), -1)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in range(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in range(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


def _njit_itakura_mask(sz1, sz2, max_slope=2.):
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = np.empty((2, sz2))
    lower_bound[0] = min_slope * np.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * np.arange(sz2))
    lower_bound_ = np.empty(sz2)
    for i in range(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, sz2))
    upper_bound[0] = max_slope * np.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * np.arange(sz2))
    upper_bound_ = np.empty(sz2)
    for i in range(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    mask = np.full((sz1, sz2), np.inf)
    for i in range(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in range(sz1):
        if not np.any(np.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in range(sz2):
            if not np.any(np.isfinite(mask[:, j])):
                raise_warning = True
                break
    return mask


if __name__ == "__main__":
    sz1, sz2 = 15, 10
    mask = sakoe_chiba_mask(sz1, sz2, radius=2)

    # Plot the warping path
    plt.close("all")
    f, ax = plt.subplots(figsize=(12, 10))
    ax = sns.heatmap(mask, vmin=-1, vmax=1, ax=ax, linewidths=0.2)
    ax.tick_params(axis="both", labelsize=12, rotation=0)
    plt.tight_layout()
