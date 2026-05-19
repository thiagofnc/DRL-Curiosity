from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from drl_curiosity.encoders import ConvEncoder, MLPEncoder


class DiscreteActorCritic(nn.Module):
    """Conv-encoder + LSTMCell + Categorical policy + scalar value (A3C-style).

    Matches the paper's policy network for pixel-based discrete-action envs
    (Doom, Mario, Atari-like). The LSTM hidden state is carried across steps
    by the trainer; call `initial_state` at episode boundaries.
    """

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 4,
        input_size: int = 42,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, input_size=input_size)
        self.lstm = nn.LSTMCell(self.encoder.output_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def initial_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self, states: Tensor, hidden: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        features = self.encoder(states)
        h, c = self.lstm(features, hidden)
        logits = self.policy(h)
        values = self.value(h).squeeze(-1)
        return logits, values, (h, c)

    def distribution(
        self, states: Tensor, hidden: tuple[Tensor, Tensor]
    ) -> tuple[Categorical, Tensor, tuple[Tensor, Tensor]]:
        logits, values, new_hidden = self.forward(states, hidden)
        return Categorical(logits=logits), values, new_hidden


class ContinuousActorCritic(nn.Module):
    """MLP-encoder + diagonal-Gaussian policy + scalar value, for continuous control.

    Gaussian std is a state-independent learnable parameter (`log_std`), which
    is the standard parameterization for PPO on MuJoCo. The distribution is
    unsquashed; the trainer is responsible for clamping samples to the action
    space (or the env enforces bounds at step time).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        encoder_out_dim: int = 64,
        initial_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            out_dim=encoder_out_dim,
        )
        self.policy = nn.Linear(encoder_out_dim, action_dim)
        self.value = nn.Linear(encoder_out_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self.action_dim = action_dim

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encoder(obs)
        mu = self.policy(features)
        log_std = self.log_std.expand_as(mu)
        value = self.value(features).squeeze(-1)
        return mu, log_std, value

    def distribution(self, obs: Tensor) -> tuple[Normal, Tensor]:
        mu, log_std, value = self.forward(obs)
        return Normal(mu, log_std.exp()), value
