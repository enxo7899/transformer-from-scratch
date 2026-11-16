"""AdamW, cosine learning-rate schedule with warmup, and global gradient clipping."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .layers import Array


def cosine_lr(step: int, total_steps: int, max_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup to max_lr, then cosine decay to max_lr * min_lr_ratio."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    floor = max_lr * min_lr_ratio
    return floor + 0.5 * (max_lr - floor) * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(grads: Iterable[Array], max_norm: float) -> float:
    """Scale all gradients in place so their global L2 norm is at most max_norm. Returns the norm before clipping."""
    grads = list(grads)
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total


class AdamW:
    """AdamW with decoupled weight decay, applied to matrices only (not biases, LayerNorm or 1-D tensors)."""

    def __init__(self, named_params: Iterable[tuple[str, Array, Array]], lr: float = 3e-3, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        self.items = [(n, p, g) for n, p, g in named_params]
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p) for _, p, _ in self.items]
        self.v = [np.zeros_like(p) for _, p, _ in self.items]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
        for (_, p, g), m, v in zip(self.items, self.m, self.v):
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            if p.ndim >= 2 and self.weight_decay:
                p *= 1.0 - self.lr * self.weight_decay
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
