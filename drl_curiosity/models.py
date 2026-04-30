from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Four-layer visual encoder used by the paper's policy and ICM."""

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


class ActorCritic(nn.Module):
    """A3C-style actor-critic network with the paper's convolutional trunk."""

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 4,
        input_size: int = 42,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(in_channels=in_channels, input_size=input_size)
        self.lstm = nn.LSTMCell(self.encoder.output_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self, states: Tensor, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        features = self.encoder(states)
        h, c = self.lstm(features, hidden)
        logits = self.policy(h)
        values = self.value(h).squeeze(-1)
        return logits, values, (h, c)


class IntrinsicCuriosityModule(nn.Module):
    """ICM from Pathak et al., using inverse and forward dynamics losses."""

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 4,
        input_size: int = 42,
        hidden_dim: int = 256,
        beta: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.beta = beta
        self.eta = eta

        self.encoder = FeatureEncoder(in_channels=in_channels, input_size=input_size)
        feature_dim = self.encoder.output_dim

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, states: Tensor, next_states: Tensor, actions: Tensor) -> dict[str, Tensor]:
        state_features = self.encoder(states)
        next_state_features = self.encoder(next_states)

        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        action_logits = self.inverse_model(inverse_input)

        action_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        forward_input = torch.cat([state_features, action_one_hot], dim=1)
        predicted_next_features = self.forward_model(forward_input)

        return {
            "state_features": state_features,
            "next_state_features": next_state_features,
            "predicted_next_features": predicted_next_features,
            "action_logits": action_logits,
        }

    def losses(self, states: Tensor, next_states: Tensor, actions: Tensor) -> dict[str, Tensor]:
        outputs = self(states, next_states, actions)
        target_next_features = outputs["next_state_features"].detach()

        inverse_loss = F.cross_entropy(outputs["action_logits"], actions)
        forward_error = 0.5 * (outputs["predicted_next_features"] - target_next_features).pow(2).sum(dim=1)
        forward_loss = forward_error.mean()
        total_loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        intrinsic_reward = self.eta * forward_error.detach()

        return {
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "total_loss": total_loss,
            "intrinsic_reward": intrinsic_reward,
        }

