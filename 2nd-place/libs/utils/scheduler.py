import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self, optimizer, max_lr=1e-3, min_lr=0.0, total=100,
        warmup=10, last_epoch=-1, verbose=False
    ):
        if (min_lr > max_lr) or (min_lr < 0.0):
            raise ValueError('min_lr expected to be between 0.0 and max_lr.')
        if (warmup < 0) or (warmup >= total):
            raise ValueError('warmup expected to be between 0 and total')

        self.min_lr     = min_lr
        self.max_lr     = max_lr
        self.total      = total
        self.warmup     = warmup
        self.num_groups = len(optimizer.param_groups)
        self.current    = None
        self.after      = total - warmup
        self.lr_range   = max_lr - min_lr

        super(WarmupCosineAnnealingLR, self).__init__(
            optimizer, last_epoch, verbose
        )

    def get_lr(self):
        if self.current is None:
            lr = 0.0
        elif self.current < self.warmup:
            lr = self.max_lr * self.current / self.warmup
        else:
            epoch_ratio = (self.current - self.warmup) / self.after
            cos_part = 1.0 + math.cos(math.pi * epoch_ratio)
            lr = self.min_lr + self.lr_range * 0.5 * cos_part
        return [lr] * self.num_groups

    def step(self, epoch=None):
        self.current = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
