"""Trains a MLP + LSTM for batch-level inference."""

import os
import time
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD

from utils.models import EmbeddingMLP
from utils.datasets import TrainDataset, EvalDataset
from utils.losses import RMSELoss

# Check if cuda is available on device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output settings
prefix = 'mlp'
version = 'a05_256'

# Train parameters
batch_size = 256
num_epochs = 5
momentum = 0.9
lr = 1e-2
margin = 1e-4
mask_pct = 0.5

# Model parameters
in_dim = 301
emb_dim = 4
hidden_dim = 256
dropout = 0.5

# Weight parameters
std_x = 1e-2
std_y = 2e-2
decay = 1e-3

# Define loss functions
criterion = RMSELoss()

# Training settings
num_workers = 4
pin_memory = True
folds = [1, 2, 3, 4, 5]
debug = False

if __name__ == '__main__':

    print('Beginning training...')
    print()

    fold_scores = []

    for fold in folds:

        # Load features
        X_train_1 = np.load(f'./data/fold_{fold}_train_feat_left.npy')
        X_train_2 = np.load(f'./data/fold_{fold}_train_feat_right.npy')
        X_valid_1 = np.load(f'./data/fold_{fold}_valid_feat_left.npy')
        X_valid_2 = np.load(f'./data/fold_{fold}_valid_feat_right.npy')

        # Loads time ids
        t_train = np.load(f'./data/fold_{fold}_train_time_id.npy')
        t_valid = np.load(f'./data/fold_{fold}_valid_time_id.npy')

        # Loads investment ids
        i_train = np.load(f'./data/fold_{fold}_train_investment_id.npy')
        i_valid = np.load(f'./data/fold_{fold}_valid_investment_id.npy')

        # Loads return targets
        y_train = np.load(f'./data/fold_{fold}_train_targets_norm.npy')
        y_valid = np.load(f'./data/fold_{fold}_valid_targets_norm.npy')

        print(f'Starting fold {fold}...')

        # Prepare debug mode
        if debug:
            X_train_1 = X_train_1[:25_000]
            X_train_2 = X_train_2[:25_000]
            X_valid_1 = X_valid_1[:25_000]
            X_valid_2 = X_valid_2[:25_000]
            y_train = y_train[:25_000]
            y_valid = y_valid[:25_000]
            t_train = t_train[:25_000]
            t_valid = t_valid[:25_000]
            i_train = i_train[:25_000]
            i_valid = i_valid[:25_000]

        # Gets number of investment_ids
        num_tokens = len(np.unique(i_train)) + 1
        tokens = sorted(list(np.unique(i_train)) + [-1])
        lookup_table = {}
        for e, i in enumerate(tokens):
            lookup_table[i] = e

        # Store available investment_ids
        np.save(f'./data/fold_{fold}_tokens.npy', tokens)

        # Mask ids not in training set
        i_train = [i if i in tokens else -1 for i in i_train]
        i_train = [lookup_table[i] for i in i_train]
        i_valid = [i if i in tokens else -1 for i in i_valid]
        i_valid = [lookup_table[i] for i in i_valid]

        train_dataset = TrainDataset(
            features_1=X_train_1, features_2=X_train_2,
            investment_ids=i_train, targets=y_train,
            std_x=std_x, std_y=std_y,
            mask_pct=mask_pct,
        )

        valid_dataset = EvalDataset(
            features_1=X_train_1, features_2=X_train_2,
            investment_ids=i_train, time_ids=t_train, targets=y_train,
        )

        test_dataset = EvalDataset(
            features_1=X_valid_1, features_2=X_valid_2,
            investment_ids=i_valid, time_ids=t_valid, targets=y_valid,
        )

        del X_train_1, X_train_2, X_valid_1, X_valid_2

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Initialize both encoder and decoder networks
        net = EmbeddingMLP(
            in_dim=in_dim, emb_dim=emb_dim, num_tokens=num_tokens,
            hidden_dim=hidden_dim, dropout=dropout,
            activation=nn.ReLU(),
        )

        # To GPU (if available)
        net = net.to(device)

        # Define optimizer and scheduler
        optimizer = SGD(
            net.parameters(), lr=lr, weight_decay=decay,
            momentum=momentum,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

        last_score = -np.inf
        old_name = '__'

        for e, epoch in enumerate(range(num_epochs)):

            # Keep track of info
            start = time.time()
            last_lr = scheduler.get_last_lr()[0]

            # Reset batch losses
            train_losses = []
            valid_losses = []
            test_losses = []

            # Keep track of metrics
            train_scores = []
            valid_scores = []
            test_scores = []

            # Training mode
            net.train()

            for b, batch in enumerate(iter(train_loader)):

                # Unpack batch
                x, i, y = batch
                x = x.to(device)
                i = i.to(device)
                y = y.to(device)

                # Reset gradient values
                optimizer.zero_grad()

                # Forward propagation through the networks
                pred = net(x, i)

                # Calculate the loss
                loss = criterion(y, pred)

                # Keep track of batch loss
                train_losses.append(loss.item())

                # Apply backpropagation
                loss.backward()

                # Update network parameters
                optimizer.step()

                # Calculate evaluation metric
                score = np.corrcoef(
                    y.detach().cpu().numpy(), pred.detach().cpu().numpy()
                )[0, 1]
                train_scores.append(score)

            # Validation mode
            net.eval()

            with torch.no_grad():
                for b, batch in enumerate(iter(valid_loader)):

                    # Unpack batch
                    x, i, y = batch
                    x = x.to(device)
                    i = i.to(device)
                    y = y.to(device)

                    # Reshaping
                    x = torch.squeeze(x)
                    i = torch.transpose(i, 1, 0)  # Squeeze, unsqueeze(dim=1)
                    y = torch.squeeze(y)

                    # Forward propagation through the networks
                    pred = net(x, i)

                    # Calculate the loss
                    loss = criterion(y, pred)

                    # Keep track of batch loss
                    valid_losses.append(loss.item())

                    # Calculate evaluation metric
                    score = np.corrcoef(
                        y.detach().cpu().numpy(), pred.detach().cpu().numpy()
                    )[0, 1]
                    valid_scores.append(score)

            with torch.no_grad():
                for b, batch in enumerate(iter(test_loader)):
                    # Unpack batch
                    x, i, y = batch
                    x = x.to(device)
                    i = i.to(device)
                    y = y.to(device)

                    # Reshaping
                    x = torch.squeeze(x)
                    i = torch.transpose(i, 1, 0)  # Squeeze, unsqueeze(dim=1)
                    y = torch.squeeze(y)

                    # Forward propagation through the networks
                    pred = net(x, i)

                    # Calculate the loss
                    loss = criterion(y, pred)

                    # Keep track of batch loss
                    test_losses.append(loss.item())

                    # Calculate evaluation metric
                    score = np.corrcoef(
                        y.detach().cpu().numpy(), pred.detach().cpu().numpy()
                    )[0, 1]
                    test_scores.append(score)

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)
            test_loss = np.mean(test_losses)

            train_score = np.mean(train_scores)
            valid_score = np.mean(valid_scores)
            test_score = np.mean(test_scores)

            # Step in learning rate scheduler
            scheduler.step()

            # Calculate duration of this epoch
            duration = int(np.round(time.time() - start))

            saved = ''

            if test_score - last_score >= margin:

                # Saves model parameters to disk
                name = f'./weights/{prefix}_{version}_{fold}_{test_score:.4f}'
                torch.save(net.state_dict(), name + '.pth')

                if os.path.exists(old_name):
                    os.remove(old_name)

                old_name = name + '.pth'
                last_score = test_score
                saved = ' - saved checkpoint!'

            print(
                f'Epoch {e + 1}/{num_epochs} - '
                f'loss {train_loss:.4f}/{valid_loss:.4f}/{test_loss:.4f} - '
                f'corr {train_score:.4f}/{valid_score:.4f}/{test_score:.4f} - '
                f'lr {last_lr:.2e} - '
                f'time {duration}s'
                f'{saved}'
            )

        # Keeps track of best scores per fold
        fold_scores.append(last_score)

        # End of fold
        print()

    print(f'Mean validation score: {np.mean(fold_scores):.4f}')
