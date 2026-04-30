from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drl_curiosity.models import ActorCritic, IntrinsicCuriosityModule


def main() -> None:
    torch.manual_seed(0)
    batch_size = 8
    action_dim = 4
    states = torch.rand(batch_size, 4, 42, 42)
    next_states = torch.rand(batch_size, 4, 42, 42)
    actions = torch.randint(0, action_dim, (batch_size,))

    policy = ActorCritic(action_dim=action_dim)
    hidden = policy.initial_state(batch_size, states.device)
    logits, values, _ = policy(states, hidden)

    icm = IntrinsicCuriosityModule(action_dim=action_dim)
    losses = icm.losses(states, next_states, actions)

    assert logits.shape == (batch_size, action_dim)
    assert values.shape == (batch_size,)
    assert losses["intrinsic_reward"].shape == (batch_size,)
    assert losses["total_loss"].isfinite()

    print("smoke test passed")
    print(f"policy logits: {tuple(logits.shape)}")
    print(f"values: {tuple(values.shape)}")
    print(f"icm loss: {losses['total_loss'].item():.4f}")
    print(f"intrinsic reward mean: {losses['intrinsic_reward'].mean().item():.4f}")


if __name__ == "__main__":
    main()
