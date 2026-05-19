from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvEncoder(nn.Module):
    """Four-layer 3x3 stride-2 ELU convolutional encoder from Pathak et al. 2017.

    Designed for stacked 42x42 grayscale frames (the paper's pixel pipeline).
    Output is a flat feature vector; `output_dim` is computed once from a dummy
    forward pass so downstream heads can size their first linear layer.
    """

    def __init__(self, in_channels: int = 4, input_size: int = 42) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            self.output_dim = self.net(dummy).shape[1]

    def forward(self, states: Tensor) -> Tensor:
        return self.net(states)


class MLPEncoder(nn.Module):
    """Tanh-MLP encoder for state-vector environments (e.g. MuJoCo HalfCheetah).

    Stable-Baselines3-style defaults: two hidden layers of 64 units with Tanh
    activations. Final tanh keeps features bounded, which plays nicely with
    the ICM forward-prediction error scale.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.output_dim = out_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)
