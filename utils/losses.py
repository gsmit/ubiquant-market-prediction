"""Collection of relevant loss functions."""

import torch

from torch import nn
from fast_soft_sort import pytorch_ops


def soft_sort(array, l2):
    return pytorch_ops.soft_sort(array.cpu(), regularization_strength=l2).cuda()


def soft_rank(array, l2):
    return pytorch_ops.soft_rank(array.cpu(), regularization_strength=l2).cuda()


def spearman_loss(pred, target, l2):
    pred = soft_rank(pred, l2=l2)
    target = soft_rank(target, l2=l2)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        rmse = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return rmse


class SpearmanWithRMSE(nn.Module):
    def __init__(self, l2=1.0, w1=1.0, w2=1.0):
        super().__init__()
        self.rmse = nn.MSELoss()
        self.spearman = spearman_loss
        self.l2 = l2
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_hat, y):
        rmse = self.rmse(y_hat, y)
        spearman = 1 - self.spearman(y_hat, y, l2=self.l2)
        loss = (self.w1 * spearman) + (self.w2 * rmse)
        return loss


class SpearmanWithBCE(nn.Module):
    def __init__(self, l2=1.0, w1=1.0, w2=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.spearman = spearman_loss
        self.l2 = l2
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_hat, y):
        spearman = 1 - self.spearman(y_hat, y, l2=self.l2)
        bce = self.bce(y_hat, y)
        loss = (self.w1 * spearman) + (self.w2 * bce)
        return loss


class CosineWithBCE(nn.Module):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_hat, y):
        cos = self.cos(
            y - y.mean(dim=1, keepdim=True),
            y_hat - y_hat.mean(dim=1, keepdim=True)
        )
        cos = 1 - torch.mean(cos)
        bce = self.bce(y_hat, y)
        loss = (self.w1 * cos) + (self.w2 * bce)
        return loss


class CosineWithRMSE(nn.Module):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.bce = RMSELoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_hat, y):
        cos = self.cos(
            y - y.mean(dim=1, keepdim=True),
            y_hat - y_hat.mean(dim=1, keepdim=True)
        )
        cos = 1 - torch.mean(cos)
        bce = self.bce(y_hat, y)
        loss = (self.w1 * cos) + (self.w2 * bce)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        mse = self.mse(x, y)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pearson = cos(
            x - x.mean(dim=1, keepdim=True), y - y.mean(dim=1, keepdim=True)
        )
        pcc = 1 - torch.mean(pearson)
        loss = (mse * self.alpha) + (pcc * self.beta)
        return loss


