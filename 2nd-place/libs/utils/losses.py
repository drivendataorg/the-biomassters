import torch
import torch.nn as nn

from piqa import SSIM


class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, prediction, target):
        diff2 = (prediction - target) ** 2
        loss = torch.sqrt(diff2 + self.eps2).mean()
        return loss


class RecLoss(nn.Module):

    def __init__(self, mode='mae', weight=1.0):
        super().__init__()

        self.weight = weight

        if mode == 'mse':
            self.loss_func = nn.MSELoss()
        elif mode == 'mae':
            self.loss_func = nn.L1Loss()
        elif mode == 'charb':
            self.loss_func = CharbonnierLoss()
        else:
            raise ValueError('unknown rec loss mode')

    def forward(self, pred, label):
        return self.loss_func(pred, label) * self.weight


class SimLoss(nn.Module):

    def __init__(self, mode='ssim', weight=1.0):
        super().__init__()

        self.weight = weight

        if mode == 'ssim':
            self.sim_func = SSIM(window_size=11, sigma=1.5, n_channels=1)
        else:
            raise ValueError('unknown sim loss mode')

    def forward(self, pred, label):
        similarity = self.sim_func(pred, label)
        sim_loss = 1.0 - similarity
        return sim_loss * self.weight

