"""Convenient helper functions for data processing."""

import numpy as np


def load_features(split, norm=True):
    """Returns the input features from a specific CV fold."""
    
    if norm:
        features_train = np.load(f'./data/fold_{split}_train_features_norm.npy')
        features_valid = np.load(f'./data/fold_{split}_valid_features_norm.npy')
    else:
        features_train = np.load(f'./data/fold_{split}_train_features.npy')
        features_valid = np.load(f'./data/fold_{split}_valid_features.npy')
        
    return features_train, features_valid


def load_targets(split):
    """Returns the targets from a specific CV fold."""

    targets_train = np.load(f'./data/fold_{split}_train_targets.npy')
    targets_valid = np.load(f'./data/fold_{split}_valid_targets.npy')

    return targets_train, targets_valid


def load_time_ids(split):
    """Returns the time ids from a specific CV fold."""

    time_id_train = np.load(f'./data/fold_{split}_train_time_id.npy')
    time_id_valid = np.load(f'./data/fold_{split}_valid_time_id.npy')

    return time_id_train, time_id_valid


def load_ranks(split, weighted=True):
    """Returns the ranks from a specific CV fold."""

    if weighted:
        ranks_train = np.load(f'./data/fold_{split}_train_rank_2.npy')
        ranks_valid = np.load(f'./data/fold_{split}_valid_rank_2.npy')
    else:
        ranks_train = np.load(f'./data/fold_{split}_train_rank_1.npy')
        ranks_valid = np.load(f'./data/fold_{split}_valid_rank_1.npy')

    return ranks_train, ranks_valid
