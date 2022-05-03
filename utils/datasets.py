"""Collection of dataset classes for different model inputs."""

import torch
import numpy as np

from torch.utils.data import Dataset
from utils.metrics import weighted_rank
from scipy.stats import rankdata


class TrainDataset(Dataset):
    """Basic dataset for the Ubiquant market prediction competition."""

    def __init__(
            self, features_1, features_2,
            investment_ids, targets,
            std_x=0.0, std_y=0.0,
            mask_pct=0.0,
    ):
        self.features_1 = torch.from_numpy(np.float32(features_1))
        self.features_2 = torch.from_numpy(np.float32(features_2))
        self.investment_ids = investment_ids
        self.targets = torch.from_numpy(np.float32(targets))
        self.std_x = std_x
        self.std_y = std_y
        self.mask_pct = mask_pct

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        features = torch.cat(
            [self.features_1[idx], self.features_2[idx]], dim=0
        )
        target = self.targets[idx]

        if np.random.random() >= self.mask_pct:
            invest_id = self.investment_ids[idx]
            invest_id = torch.LongTensor([invest_id])
        else:
            invest_id = torch.LongTensor([0])

        if self.std_x > 0:
            shape = features.shape
            noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_x)
            features = features + noise

        if self.std_y > 0:
            shape = target.shape
            noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_y)
            target = target + noise

        return features, invest_id, target


class EvalDataset(Dataset):
    """Dataset that feeds all observations of a single time_id per batch."""

    def __init__(
            self, features_1, features_2,
            investment_ids, time_ids, targets,
    ):
        self.features_1 = torch.from_numpy(np.float32(features_1))
        self.features_2 = torch.from_numpy(np.float32(features_2))
        self.investment_ids = np.int64(investment_ids)
        self.targets = torch.from_numpy(np.float32(targets))
        self.time_ids = time_ids

        self.unique = np.unique(self.time_ids)
        self.indices = np.arange(len(self.targets))
        self.lookup = {}

        for time_id in self.unique:
            self.lookup[time_id] = self.indices[(self.time_ids == time_id)]

    def __len__(self):
        return len(self.unique)

    def __getitem__(self, idx):
        time_id = self.unique[idx]
        indices = self.lookup[time_id]

        features = torch.concat(
            [self.features_1[indices], self.features_2[indices]], dim=1
        )
        targets = self.targets[indices]
        invest_ids = self.investment_ids[indices]
        invest_ids = torch.LongTensor(invest_ids)

        return features, invest_ids, targets


class BatchDatasetV2(Dataset):
    """Dataset that feeds all observations of a single time_id per batch."""

    def __init__(
            self, features_1, features_2, targets, time_ids, augment=True,
            shuffle=True, std_x=0.0, std_y=0.0, weight=0.5,
            sample=0.8, scale=True,
    ):
        assert features_1.shape[0] == time_ids.shape[0]
        self.features_1 = torch.from_numpy(features_1)
        self.features_2 = torch.from_numpy(features_2)
        self.targets = torch.from_numpy(targets)
        self.time_ids = time_ids
        self.augment = augment
        self.shuffle = shuffle
        self.std_x = std_x
        self.std_y = std_y
        self.sample = sample

        self.scale = scale

        self.unique = np.unique(self.time_ids)
        self.indices = np.arange(len(self.targets))
        self.lookup = {}

        for time_id in self.unique:
            self.lookup[time_id] = self.indices[(self.time_ids == time_id)]

    def __len__(self):
        return len(self.unique)

    def __getitem__(self, idx):
        time_id = self.unique[idx]
        indices = self.lookup[time_id]
        features = torch.concat(
            [self.features_1[indices], self.features_2[indices]], dim=1
        )
        # features = np.concatenate(
        #     [self.features_1[indices], self.features_2[indices]], axis=1
        # )
        targets = self.targets[indices]

        # Data augmentation
        if self.augment:
            n_samples = np.random.randint(
                low=int(round(self.sample * len(indices))),
                high=len(indices) + 1
            )

            subset = np.random.choice(
                np.arange(len(features)), size=n_samples,
                replace=False
            )

            features = features[subset]
            targets = targets[subset]

        # if self.std_x > 0:
        #     shape = features.shape
        #     noise = np.random.normal(loc=0.0, scale=self.std_x, size=shape)
        #     features = features + noise
        #
        # if self.std_y > 0:
        #     shape = targets.shape
        #     noise = np.random.normal(loc=0.0, scale=self.std_y, size=shape)
        #     targets = np.clip(targets + noise, a_min=-1.0, a_max=1.0)

        if self.std_x > 0:
            shape = features.shape
            noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_x)
            features = features + noise

        if self.std_y > 0:
            shape = targets.shape
            noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_y)
            targets = targets + noise

        if self.scale:
            # features = self.scaler.fit_transform(features)
            # features = (features - np.mean(features) / np.std(features))
            pass

        # Data augmentation
        if self.shuffle:
            shuffle_idx = np.random.permutation(len(features))
            features = features[shuffle_idx]
            targets = targets[shuffle_idx]

        # To tensor format
        features = features.to(torch.float32)
        targets = targets.to(torch.float32)

        return features, targets


class BatchDataset(Dataset):
    """Dataset that feeds all observations of a single time_id per batch."""

    def __init__(
            self, features_1, features_2, targets, time_ids, augment=True,
            shuffle=True, std_x=0.0, std_y=0.0, weight=0.5,
            sample=0.8, scale=True,
    ):
        assert features_1.shape[0] == time_ids.shape[0]
        self.features_1 = np.float32(features_1)
        self.features_2 = np.float32(features_2)
        self.targets = targets
        self.time_ids = time_ids
        self.augment = augment
        self.shuffle = shuffle
        self.std_x = std_x
        self.std_y = std_y
        self.weight = weight
        self.sample = sample

        self.scale = scale

        self.unique = np.unique(self.time_ids)
        self.indices = np.arange(len(self.targets))
        self.lookup = {}

        for time_id in self.unique:
            self.lookup[time_id] = self.indices[(self.time_ids == time_id)]

    def __len__(self):
        return len(self.unique)

    def __getitem__(self, idx):
        time_id = self.unique[idx]
        indices = self.lookup[time_id]
        # features = torch.concat(
        #     [self.features_1[indices], self.features_2[indices]], dim=1
        # )
        features = np.concatenate(
            [self.features_1[indices], self.features_2[indices]], axis=1
        )
        targets = np.float32(self.targets[indices])

        # Data augmentation
        if self.augment:
            n_samples = np.random.randint(
                low=int(round(self.sample * len(indices))),
                high=len(indices) + 1
            )

            subset = np.random.choice(
                np.arange(len(features)), size=n_samples,
                replace=False
            )

            features = features[subset]
            targets = targets[subset]

        if self.std_x > 0:
            shape = features.shape
            noise = np.random.normal(loc=0.0, scale=self.std_x, size=shape)
            features = features + noise

        if self.std_y > 0:
            shape = targets.shape
            noise = np.random.normal(loc=0.0, scale=self.std_y, size=shape)
            targets = np.clip(targets + noise, a_min=-1.0, a_max=1.0)

        if self.scale:
            # features = self.scaler.fit_transform(features)
            # features = (features - np.mean(features) / np.std(features))
            pass

        # Convert targets to ranks
        ranks1 = rankdata(targets, method='ordinal') - 1
        ranks1 = np.float32(ranks1 / (len(ranks1) - 1))
        ranks2 = (
                (targets - np.min(targets)) /
                (np.max(targets) - np.min(targets))
        )
        targets = (self.weight * ranks1) + ((1.0 - self.weight) * ranks2)

        # Data augmentation
        if self.shuffle:
            shuffle_idx = np.random.permutation(len(features))
            features = features[shuffle_idx]
            targets = targets[shuffle_idx]

        # To tensor format
        features = torch.from_numpy(features).to(torch.float32)
        targets = torch.from_numpy(targets).to(torch.float32)

        # if self.std_x > 0:
        #     shape = features.shape
        #     noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_x)
        #     features = features + noise
        #
        # if self.std_y > 0:
        #     shape = targets.shape
        #     noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std_y)
        #     targets = torch.clip(targets + noise, min=-1.0, max=1.0)

        return features, targets


class PairwiseRankingDatasetV3(Dataset):
    """Dataset that feeds pairs of samples with a binary rank."""

    def __init__(
            self, features_path, targets_path,
            time_id_path, n_samples=128, repeats=1,
    ):
        self.features = torch.from_numpy(np.float16(np.load(features_path)))
        self.targets = np.float32(np.load(targets_path))
        self.time_id = np.float32(np.load(time_id_path))
        self.n_samples = n_samples

        self.unique = np.sort(np.unique(self.time_id))
        self.unique = np.repeat(self.unique, repeats=repeats)

        self.indices = np.arange(len(self.targets))
        self.lookup = {}

        for time_id in self.unique:
            self.lookup[time_id] = self.indices[(self.time_id == time_id)]

        assert self.features.shape[0] == len(self.targets) == len(self.time_id)

        # TODO: Implement combinatorial order of samples.
        pass

    def __len__(self):
        return len(self.unique)

    def __getitem__(self, idx):

        n_samples = self.n_samples
        time_id = self.unique[idx]
        indices = self.lookup[time_id]
        samples = np.random.choice(indices, size=n_samples, replace=False)

        features = self.features[samples]
        targets = self.targets[samples]

        ranks = weighted_rank(targets, weight=0.2)
        ranks = torch.from_numpy(ranks)

        features = features.to(torch.float32)
        ranks = ranks.to(torch.float32)

        noise = torch.zeros(size=features.shape).normal_(mean=0.0, std=1e-3)
        features = features + noise

        return features, ranks


class AutoencoderDataset(Dataset):
    """Dataset for reconstruction of the input (with optional noise)."""

    def __init__(self, features_path, targets_path, indices, std=1e-4):
        self.features_path = features_path
        self.targets_path = targets_path
        self.index = indices
        self.std = std

        self.features = torch.from_numpy(np.load(features_path)[indices])
        self.targets = torch.from_numpy(np.load(targets_path)[indices])

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]

        if self.std > 0:
            shape = features.shape
            noise = torch.zeros(size=shape).normal_(mean=0.0, std=self.std)
            features = features + noise

        return features, targets
