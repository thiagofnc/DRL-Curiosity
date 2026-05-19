from __future__ import annotations

import argparse
from collections import deque

import torch
from torch.distributions import Categorical

from drl_curiosity.encoders import ConvEncoder
from drl_curiosity.icm import DiscreteICM
from drl_curiosity.policies import DiscreteActorCritic
from drl_curiosity.preprocessing import preprocess_frame, stack_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ICM + actor-critic agent.")
    parser.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--icm-loss-coef", type=float, default=1.0)
    parser.add_argument("--intrinsic-reward-scale", type=float, default=0.01)
    return parser.parse_args()


def make_env(env_id: str):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Gymnasium is required for training. Install it with `pip install gymnasium` "
            "plus the package for your chosen environment."
        ) from exc
    return gym.make(env_id)


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action: int):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = result
    return obs, reward, done, info


def build_state(frame_buffer: deque[torch.Tensor]) -> torch.Tensor:
    return stack_frames(list(frame_buffer), stack_size=4)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id)

    if not hasattr(env.action_space, "n"):
        raise SystemExit("This starter implementation expects a discrete action space.")

    action_dim = env.action_space.n
    policy = DiscreteActorCritic(action_dim=action_dim).to(device)
    icm = DiscreteICM(
        feature_encoder=ConvEncoder().to(device),
        action_dim=action_dim,
        eta=args.intrinsic_reward_scale,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(icm.parameters()),
        lr=args.lr,
    )

    obs = reset_env(env)
    first_frame = preprocess_frame(obs).to(device)
    frame_buffer: deque[torch.Tensor] = deque([first_frame] * 4, maxlen=4)
    state = build_state(frame_buffer)
    hidden = policy.initial_state(batch_size=1, device=device)
    episode_return = 0.0
    episode_count = 0

    for global_step in range(0, args.total_steps, args.rollout_steps):
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        dones: list[bool] = []
        states: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        for _ in range(args.rollout_steps):
            logits, value, hidden = policy(state.unsqueeze(0), hidden)
            dist = Categorical(logits=logits)
            action = dist.sample()

            extrinsic_reward = 0.0
            done = False
            obs = None
            for _ in range(args.action_repeat):
                obs, reward, done, _ = step_env(env, int(action.item()))
                extrinsic_reward += float(reward)
                if done:
                    break

            next_frame = preprocess_frame(obs).to(device)
            frame_buffer.append(next_frame)
            next_state = build_state(frame_buffer)

            with torch.no_grad():
                intrinsic_reward = icm.losses(
                    state.unsqueeze(0),
                    next_state.unsqueeze(0),
                    action,
                )["intrinsic_reward"].item()

            log_probs.append(dist.log_prob(action).squeeze(0))
            values.append(value.squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            rewards.append(extrinsic_reward + intrinsic_reward)
            dones.append(done)
            states.append(state)
            next_states.append(next_state)
            actions.append(action.squeeze(0))

            episode_return += extrinsic_reward
            state = next_state

            if done:
                episode_count += 1
                print(
                    f"step={global_step:>8} episode={episode_count:>4} "
                    f"extrinsic_return={episode_return:.2f}"
                )
                obs = reset_env(env)
                first_frame = preprocess_frame(obs).to(device)
                frame_buffer = deque([first_frame] * 4, maxlen=4)
                state = build_state(frame_buffer)
                hidden = policy.initial_state(batch_size=1, device=device)
                episode_return = 0.0

        with torch.no_grad():
            if dones[-1]:
                next_value = torch.zeros((), device=device)
            else:
                _, bootstrap_value, _ = policy(state.unsqueeze(0), hidden)
                next_value = bootstrap_value.squeeze(0)

            returns: list[torch.Tensor] = []
            running_return = next_value
            for reward, done in zip(reversed(rewards), reversed(dones)):
                running_return = torch.tensor(reward, device=device) + args.gamma * running_return * (not done)
                returns.insert(0, running_return)

        returns_tensor = torch.stack(returns)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        advantages = returns_tensor - values_tensor

        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()
        entropy_loss = entropies_tensor.mean()

        states_tensor = torch.stack(states)
        next_states_tensor = torch.stack(next_states)
        actions_tensor = torch.stack(actions)
        icm_loss = icm.losses(states_tensor, next_states_tensor, actions_tensor)["total_loss"]

        loss = (
            policy_loss
            + args.value_loss_coef * value_loss
            - args.entropy_coef * entropy_loss
            + args.icm_loss_coef * icm_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(icm.parameters()), max_norm=40.0)
        optimizer.step()
        hidden = (hidden[0].detach(), hidden[1].detach())

        print(
            f"step={global_step + args.rollout_steps:>8} "
            f"loss={loss.item():.3f} policy={policy_loss.item():.3f} "
            f"value={value_loss.item():.3f} icm={icm_loss.item():.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()

