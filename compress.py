"""Splits and converts data to a more efficient format."""

import numpy as np
import pandas as pd

base_cols = ['time_id', 'investment_id', 'target']
features = [f'f_{i}' for i in range(300)]

dtypes = {
    'time_id': 'uint16',
    'investment_id': 'uint16',
    'target': 'float32',
}

for col in features:
    dtypes[col] = 'float32'

print('Loading csv...')

columns = base_cols + features
df = pd.read_csv(
    'input/train.csv',
    usecols=columns,
    dtype=dtypes
)

df['rank1'] = df.groupby('time_id')['target'].rank(pct=True)
target_min = df.groupby('time_id')['target'].min()
target_max = df.groupby('time_id')['target'].max()
df['rank2'] = (df['target'] - target_min) / (target_max - target_min)

print('Finalizing datasets...')
time_id = df.pop('time_id').values
investment_id = df.pop('investment_id').values
target = df.pop('target').values
rank1 = df.pop('rank1').values
rank2 = df.pop('rank2').values

with open('input/time_id.npy', 'wb') as f:
    np.save(f, time_id)

with open('input/investment_id.npy', 'wb') as f:
    np.save(f, investment_id)

with open('input/target.npy', 'wb') as f:
    np.save(f, target)

with open('input/rank1.npy', 'wb') as f:
    np.save(f, rank1)

with open('input/rank2.npy', 'wb') as f:
    np.save(f, rank2)

del time_id, investment_id, target, rank1, rank2

df = np.float32(df.values)

with open('input/data.npy', 'wb') as f:
    np.save(f, df)

print('Done!')
