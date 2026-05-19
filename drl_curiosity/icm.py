from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _ICMBase(nn.Module):
    """Shared scaffolding for ICM variants.

    The feature encoder is injected so the trainer can choose whether the ICM
    shares its encoder with the policy (not recommended -- see the paper) or
    keeps its own. The paper's design uses a *separate* encoder so that the
    inverse model alone is responsible for shaping the curiosity feature
    space; that is what trainers should construct by default.
    """

    def __init__(
        self,
        feature_encoder: nn.Module,
        action_dim: int,
        hidden_dim: int,
        beta: float,
        eta: float,
        inverse_input_dim: int,
        forward_input_dim: int,
    ) -> None:
        super().__init__()
        if not hasattr(feature_encoder, "output_dim"):
            raise ValueError(
                "feature_encoder must expose an `output_dim` attribute "
                "(see encoders.ConvEncoder / encoders.MLPEncoder)."
            )
        self.encoder = feature_encoder
        self.action_dim = action_dim
        self.beta = beta
        self.eta = eta

        feature_dim = self.encoder.output_dim
        self.inverse_model = nn.Sequential(
            nn.Linear(inverse_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(forward_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )


class DiscreteICM(_ICMBase):
    """ICM for discrete action spaces (paper-faithful variant).

    - Inverse model: classifier over actions, trained with cross-entropy.
    - Forward model: consumes the one-hot action, predicts next features.
    - Intrinsic reward: 0.5 * ||phi_hat(s') - phi(s')||^2, scaled by eta.
    """

    def __init__(
        self,
        feature_encoder: nn.Module,
        action_dim: int,
        hidden_dim: int = 256,
        beta: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        feature_dim = feature_encoder.output_dim
        super().__init__(
            feature_encoder=feature_encoder,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            eta=eta,
            inverse_input_dim=feature_dim * 2,
            forward_input_dim=feature_dim + action_dim,
        )

    def forward(
        self, states: Tensor, next_states: Tensor, actions: Tensor
    ) -> dict[str, Tensor]:
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

    def losses(
        self, states: Tensor, next_states: Tensor, actions: Tensor
    ) -> dict[str, Tensor]:
        outputs = self(states, next_states, actions)
        target_next_features = outputs["next_state_features"].detach()

        inverse_loss = F.cross_entropy(outputs["action_logits"], actions)
        forward_error = 0.5 * (
            outputs["predicted_next_features"] - target_next_features
        ).pow(2).sum(dim=1)
        forward_loss = forward_error.mean()
        total_loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        intrinsic_reward = self.eta * forward_error.detach()

        return {
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "total_loss": total_loss,
            "intrinsic_reward": intrinsic_reward,
            "forward_error": forward_error.detach(),
        }


class ContinuousICM(_ICMBase):
    """ICM for continuous action spaces (used by the PPO / MuJoCo trainer).

    - Inverse model: regresses action vectors with MSE.
    - Forward model: takes the raw action vector (no one-hot), predicts
      next features.
    - Intrinsic reward: same form as the discrete case.
    """

    def __init__(
        self,
        feature_encoder: nn.Module,
        action_dim: int,
        hidden_dim: int = 256,
        beta: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        feature_dim = feature_encoder.output_dim
        super().__init__(
            feature_encoder=feature_encoder,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            eta=eta,
            inverse_input_dim=feature_dim * 2,
            forward_input_dim=feature_dim + action_dim,
        )

    def forward(
        self, states: Tensor, next_states: Tensor, actions: Tensor
    ) -> dict[str, Tensor]:
        state_features = self.encoder(states)
        next_state_features = self.encoder(next_states)

        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        predicted_actions = self.inverse_model(inverse_input)

        forward_input = torch.cat([state_features, actions], dim=1)
        predicted_next_features = self.forward_model(forward_input)

        return {
            "state_features": state_features,
            "next_state_features": next_state_features,
            "predicted_next_features": predicted_next_features,
            "predicted_actions": predicted_actions,
        }

    def losses(
        self, states: Tensor, next_states: Tensor, actions: Tensor
    ) -> dict[str, Tensor]:
        outputs = self(states, next_states, actions)
        target_next_features = outputs["next_state_features"].detach()

        inverse_loss = F.mse_loss(outputs["predicted_actions"], actions)
        forward_error = 0.5 * (
            outputs["predicted_next_features"] - target_next_features
        ).pow(2).sum(dim=1)
        forward_loss = forward_error.mean()
        total_loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        intrinsic_reward = self.eta * forward_error.detach()

        return {
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "total_loss": total_loss,
            "intrinsic_reward": intrinsic_reward,
            "forward_error": forward_error.detach(),
        }
