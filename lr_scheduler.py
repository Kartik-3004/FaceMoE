from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import SGD
import torch
import warnings


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


class PolynomialLRWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:        
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = ((1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):

        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return [
                (
                    base_lr * (1.0 - (min(self.total_iters, self.last_epoch) - self.warmup_iters) / (self.total_iters - self.warmup_iters)) ** self.power
                )
                for base_lr in self.base_lrs
            ]
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.num_epoch * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epoch * n_iter_per_epoch)
    decay_steps = int(config.decay_epoch * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in []]

    lr_scheduler = None
    if config.scheduler== 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps),
            t_mul=1.,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
        )
    elif config.scheduler== 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.scheduler == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        
        assert self.warmup_t <= min(self.milestones)
    
    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
    
if __name__ == "__main__":

    class TestModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
        
        def forward(self, x):
            return self.linear(x)

    test_module = TestModule()
    test_module_pfc = TestModule()
    lr_pfc_weight = 1 / 3
    base_lr = 10
    total_steps = 1000
    
    sgd = SGD([
        {"params": test_module.parameters(), "lr": base_lr},
        {"params": test_module_pfc.parameters(), "lr": base_lr * lr_pfc_weight}
        ], base_lr)

    scheduler = PolynomialLRWarmup(sgd, total_steps//10, total_steps, power=2)

    x = []
    y = []
    y_pfc = []
    for i in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lr_pfc = scheduler.get_last_lr()[1]
        x.append(i)
        y.append(lr)
        y_pfc.append(lr_pfc)

    import matplotlib.pyplot as plt
    fontsize=15
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, linestyle='-', linewidth=2, )
    plt.plot(x, y_pfc, linestyle='-', linewidth=2, )
    plt.xlabel('Iterations')     # x_label
    plt.ylabel("Lr")             # y_label
    plt.savefig("tmp.png", dpi=600, bbox_inches='tight')
