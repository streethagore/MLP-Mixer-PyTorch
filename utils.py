# Description: Contains utility functions and classes

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Optimizer
def get_optimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

# Learning rate scheduler
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_batches_per_epoch, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            current_step = self.current_step + self.current_epoch * self.num_batches_per_epoch
            lr = self.base_lr * (current_step / (self.warmup_epochs * self.num_batches_per_epoch))
        else:
            progress = ((self.current_step - self.warmup_epochs * self.num_batches_per_epoch) /
                        ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0

# AverageMeter class to keep track of losses and accuracies
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def average(self):
        return self.sum / self.count

# CutMix function
def cutmix_data(x, y, beta=1.0, cutmix_prob=0.5):
    if np.random.rand() > cutmix_prob:
        return x, y, y, 1.0  # No CutMix applied

    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    return x, y, shuffled_y, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2