from __future__ import annotations

import torch
from torch import Tensor


class RunningMeanStd:
    """Welford / Chan-style parallel running mean and variance.

    Lightweight, pure-torch, no autograd. Used for two purposes in this repo:

    1. Observation normalization on MuJoCo (subtract mean, divide by std,
       clip to +/-10).
    2. Intrinsic-reward normalization for ICM: maintain a running estimate of
       the std of *intrinsic returns* (discounted sum of forward errors)
       across rollouts, then divide raw per-step forward errors by that std
       before applying eta. This is the RND-style normalization recipe that
       makes curiosity scale comparable across envs and across training time.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = float(epsilon)

    def update(self, x: Tensor) -> None:
        """Update stats from a batch `x` of shape (N, *shape)."""
        x = x.detach().to(self.mean.device).float()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total_count)
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> Tensor:
        return self.var.sqrt()

    def normalize(
        self, x: Tensor, eps: float = 1e-8, subtract_mean: bool = True
    ) -> Tensor:
        """Standardize `x`. Pass subtract_mean=False to only divide by std
        (the right choice for intrinsic-reward normalization)."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        if subtract_mean:
            return (x - mean) / (std + eps)
        return x / (std + eps)
