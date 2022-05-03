"""Creates splits of the data and applies normalization."""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer

base_cols = ['time_id', 'investment_id', 'target']
features = [f'f_{i}' for i in range(300)]
remove_outliers = False

dtypes = {
    'time_id': 'int32',
    'investment_id': 'int32',
    'target': 'float32',
}

for col in features:
    dtypes[col] = 'float16'

print('Loading csv...')

columns = base_cols + features

df = pd.read_csv(
    'input/train.csv',
    usecols=columns,
    dtype=dtypes
)

if remove_outliers:
    df = df.loc[(df['time_id'] < 350) | (df['time_id'] >= 550)]

df['rank1'] = df.groupby('time_id')['target'].rank(pct=True)
target_min = df.groupby('time_id')['target'].transform('min')
target_max = df.groupby('time_id')['target'].transform('max')
df['rank2'] = (df['target'] - target_min) / (target_max - target_min)

# Split dataframe in arrays
time_id = df.pop('time_id').values
investment_id = df.pop('investment_id').values
targets = df.pop('target').values
rank1 = np.float32(df.pop('rank1').values)
rank2 = np.float32(df.pop('rank2').values)
train = np.float16(df.values)

print(f'Prepared features and targets...')
print(np.sum(np.isnan(rank1)))
print(np.sum(np.isnan(rank2)))

# Normalize input features
scaler = QuantileTransformer(
    n_quantiles=1000,
    output_distribution='normal',
    subsample=1_000_000,
    random_state=42,
    copy=False,
)

norm = scaler.fit_transform(train)
norm = np.float16(norm)
joblib.dump(scaler, f'./data/quantile_transformer.pkl', compress=True)

print(f'Normalized input features...')

# Create multiple data splits
kfold = GroupKFold(n_splits=5)

for e, (train_idx, test_idx) in enumerate(kfold.split(time_id, time_id, time_id)):

    # Current fold number
    split = e + 1

    print(f'Processing fold {split}...')

    # Separate data into train and test splits
    features_train, features_test = train[train_idx], train[test_idx]
    targets_train, targets_test = targets[train_idx], targets[test_idx]
    time_id_train, time_id_test = time_id[train_idx], time_id[test_idx]
    rank_1_train, rank_1_test = rank1[train_idx], rank1[test_idx]
    rank_2_train, rank_2_test = rank2[train_idx], rank2[test_idx]

    # Save data to disk
    np.save(f'./data/fold_{split}_train_rank_1.npy', rank_1_train)
    np.save(f'./data/fold_{split}_valid_rank_1.npy', rank_1_test)
    np.save(f'./data/fold_{split}_train_rank_2.npy', rank_2_train)
    np.save(f'./data/fold_{split}_valid_rank_2.npy', rank_2_test)
    np.save(f'./data/fold_{split}_train_time_id.npy', time_id_train)
    np.save(f'./data/fold_{split}_valid_time_id.npy', time_id_test)
    np.save(f'./data/fold_{split}_train_targets.npy', targets_train)
    np.save(f'./data/fold_{split}_valid_targets.npy', targets_test)
    np.save(f'./data/fold_{split}_train_features.npy', features_train)
    np.save(f'./data/fold_{split}_valid_features.npy', features_test)
    del time_id_train, time_id_test, targets_train, targets_test

    # Feature normalization
    norm_train, norm_test = norm[train_idx], norm[test_idx]
    np.save(f'./data/fold_{split}_train_normalized.npy', norm_train)
    np.save(f'./data/fold_{split}_valid_normalized.npy', norm_test)
    del features_train, features_test

    print(f'Saved data to disk...')

    print(f'Finished processing fold {split}!')
    print()
