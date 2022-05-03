"""Task specific functions for model inference."""

import itertools
import numpy as np


def get_permutations(groups, seq_size=256):
    """Get all permutations of multiple time id group."""

    indices = np.arange(len(groups))
    unique_groups = np.sort(np.unique(groups))
    group_combinations = []

    for time_id in unique_groups:
        time_indices = indices[(groups == time_id)]
        combinations = list(itertools.combinations(time_indices, r=seq_size))
        group_combinations += combinations

    return group_combinations
