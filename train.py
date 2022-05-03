"""Trains a MLP + LSTM for batch-level inference."""

import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam, SGD

from utils.models import MultilayerRNN
from utils.datasets import BatchDataset
from utils.losses import SpearmanWithRMSE

# Check if cuda is available on device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
batch_size = 1
num_epochs = 10
momentum = 0.9
lr = 1e-4
h_dim = 128
z_dim = 4
dropout = 0.375
margin = 1e-3
scale = False

# Output settings
prefix = 'lstm'
version = 'a03'

# Weight parameters
w_feat = 1.0
w_rank = 0.8
w_rmse = 0.5
sample = 0.9
std_x = 1e-2
std_y = 1e-2
decay = 1e-4
l2 = 1.0

# Training settings
num_workers = 0
pin_memory = True
folds = [9]
debug = False

if __name__ == '__main__':

    print('Beginning training...')
    print()

    for fold in folds:

        # Loads time ids
        t_train = np.load(f'./data/fold_{fold}_train_time_id.npy')
        t_valid = np.load(f'./data/fold_{fold}_valid_time_id.npy')

        # Loads return targets
        y_train = np.load(f'./data/fold_{fold}_train_targets.npy')
        y_valid = np.load(f'./data/fold_{fold}_valid_targets.npy')

        # Loads and weights input features
        X_train = np.load(f'./data/fold_{fold}_train_features.npy')
        X_valid = np.load(f'./data/fold_{fold}_valid_features.npy')

        print(f'Starting fold {fold}...')

        # Prepare debug mode
        if debug:
            X_train = X_train[:10_000]
            X_valid = X_valid[:10_000]
            y_train = y_train[:10_000]
            y_valid = y_valid[:10_000]
            t_train = t_train[:10_000]
            t_valid = t_valid[:10_000]

        train_dataset = BatchDataset(
            X_train, y_train, t_train, augment=True, shuffle=True,
            std_x=std_x, std_y=std_y, weight=w_rank, sample=sample, scale=scale,
        )

        valid_dataset = BatchDataset(
            X_valid, y_valid, t_valid, augment=False, shuffle=False,
            std_x=0.0, std_y=0.0, weight=w_rank, sample=1.0, scale=scale,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Initialize both encoder and decoder networks
        net = MultilayerRNN(
            in_dim=300, hidden_dim=h_dim,
            emb_dim=z_dim, dropout=dropout,
        )

        # To GPU (if available)
        net = net.to(device)

        # Define optimizer and scheduler
        optimizer = SGD(
            net.parameters(), lr=lr, weight_decay=decay,
            momentum=momentum,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

        # Define loss functions
        criterion = SpearmanWithRMSE(l2=l2, w1=1.0, w2=w_rmse)

        last_score = -np.inf
        old_name = '__'

        for e, epoch in enumerate(range(num_epochs)):

            # Keep track of info
            start = time.time()
            last_lr = scheduler.get_last_lr()[0]

            # Reset batch losses
            train_losses = []
            valid_losses = []

            # Keep track of metrics
            train_scores = []
            valid_scores = []

            # Training mode
            net.train()

            for b, batch in enumerate(iter(train_loader)):

                # Unpack batch
                x, r = batch
                x = x.to(device)
                r = r.to(device)

                # Reset gradient values
                optimizer.zero_grad()

                # Forward propagation through the networks
                pred = net(x)

                # Calculate the loss
                loss = criterion(r, pred)

                # Keep track of batch loss
                train_losses.append(loss.item())

                # Apply backpropagation
                loss.backward()

                # Update network parameters
                optimizer.step()

                # Calculate evaluation metric
                score = np.corrcoef(
                    r.detach().cpu().numpy(), pred.detach().cpu().numpy()
                )[0, 1]
                train_scores.append(score)

            # Validation mode
            net.eval()

            with torch.no_grad():
                for b, batch in enumerate(iter(valid_loader)):
                    # Unpack batch
                    x, r = batch
                    x = x.to(device)
                    r = r.to(device)

                    # Forward propagation through the networks
                    pred = net(x)

                    # Calculate the loss
                    loss = criterion(r, pred)

                    # Keep track of batch loss
                    valid_losses.append(loss.item())

                    # Calculate evaluation metric
                    score = np.corrcoef(
                        r.detach().cpu().numpy(), pred.detach().cpu().numpy()
                    )[0, 1]
                    valid_scores.append(score)

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)
            train_score = np.mean(train_scores)
            valid_score = np.mean(valid_scores)

            # Step in learning rate scheduler
            scheduler.step()

            # Calculate duration of this epoch
            duration = int(np.round(time.time() - start))

            saved = ''

            if valid_score - last_score >= margin:

                # Saves model parameters to disk
                name = f'./weights/{prefix}_{version}_{fold}_{valid_score:.4f}'
                torch.save(net.state_dict(), name + '.pth')

                if os.path.exists(old_name):
                    os.remove(old_name)

                old_name = name + '.pth'
                last_score = valid_score
                saved = ' - saved checkpoint!'

            print(
                f'Epoch {e + 1}/{num_epochs} - '
                f'loss {train_loss:.4f}/{valid_loss:.4f} - '
                f'corr {train_score:.4f}/{valid_score:.4f} - '
                f'lr {last_lr:.2e} - '
                f'time {duration}s'
                f'{saved}'
            )

        # End
        print()
