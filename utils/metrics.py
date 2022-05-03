"""Evaluation metrics for this specific project."""

import torch
import numpy as np
import pandas as pd

from scipy.stats import rankdata


def weighted_rank(targets, weight=1.0):
    """Returns an array of weighted ranks given some targets."""

    ranks = rankdata(targets)
    ranks = (ranks - 1) / (len(ranks) - 1)

    min_val = np.min(targets)
    max_val = np.max(targets)
    norm_ranks = (targets - min_val) / (max_val - min_val)

    result = ((1 - weight) * ranks) + (weight * norm_ranks)
    result = (result * 2) - 1  # Scale to [-1, 1]
    result = np.float32(result)

    return result


def accuracy(y_prob, y_true):
    y_prob = torch.round(y_prob)
    return (y_true == y_prob).sum() / y_true.shape[0]


def pearson_coef(data):
    return data.corr()['target']['preds']


def eval_metric(valid_df):
    cols = ['time_id', 'target', 'preds']
    coef = valid_df[cols].groupby('time_id').apply(pearson_coef)
    mean = np.mean(coef)
    return mean


def corrcoef(arr1, arr2):
    """Returns the Pearson correlation coefficient."""

    vx = arr1 - torch.mean(arr1)
    vy = arr2 - torch.mean(arr2)
    cost = (torch.sum(vx * vy) /
            (torch.sqrt(torch.sum(vx ** 2)) *
             torch.sqrt(torch.sum(vy ** 2)))
            )
    return cost
