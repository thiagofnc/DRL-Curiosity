from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drl_curiosity.encoders import ConvEncoder, MLPEncoder
from drl_curiosity.icm import ContinuousICM, DiscreteICM
from drl_curiosity.policies import ContinuousActorCritic, DiscreteActorCritic
from drl_curiosity.running_stats import RunningMeanStd


def _check_discrete_pixel_path() -> None:
    """Original paper path: stacked frames, conv encoder, LSTM policy, discrete ICM."""
    batch_size = 8
    action_dim = 4
    states = torch.rand(batch_size, 4, 42, 42)
    next_states = torch.rand(batch_size, 4, 42, 42)
    actions = torch.randint(0, action_dim, (batch_size,))

    policy = DiscreteActorCritic(action_dim=action_dim)
    hidden = policy.initial_state(batch_size, states.device)
    logits, values, _ = policy(states, hidden)

    icm = DiscreteICM(
        feature_encoder=ConvEncoder(),
        action_dim=action_dim,
    )
    losses = icm.losses(states, next_states, actions)

    assert logits.shape == (batch_size, action_dim)
    assert values.shape == (batch_size,)
    assert losses["intrinsic_reward"].shape == (batch_size,)
    assert losses["total_loss"].isfinite()
    print("[discrete/pixel] OK"
          f" | logits={tuple(logits.shape)} values={tuple(values.shape)}"
          f" icm_loss={losses['total_loss'].item():.4f}"
          f" intrinsic_mean={losses['intrinsic_reward'].mean().item():.4f}")


def _check_continuous_state_path() -> None:
    """HalfCheetah-style path: state vectors, MLP encoder, Gaussian policy, continuous ICM."""
    batch_size = 8
    obs_dim = 17  # HalfCheetah observation dim
    action_dim = 6  # HalfCheetah action dim
    obs = torch.randn(batch_size, obs_dim)
    next_obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim).clamp(-1, 1)

    policy = ContinuousActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    dist, values = policy.distribution(obs)
    sampled = dist.sample()
    log_probs = dist.log_prob(sampled).sum(-1)

    icm = ContinuousICM(
        feature_encoder=MLPEncoder(obs_dim=obs_dim),
        action_dim=action_dim,
    )
    losses = icm.losses(obs, next_obs, actions)

    assert sampled.shape == (batch_size, action_dim)
    assert values.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert losses["intrinsic_reward"].shape == (batch_size,)
    assert losses["total_loss"].isfinite()
    print("[continuous/state] OK"
          f" | action={tuple(sampled.shape)} values={tuple(values.shape)}"
          f" icm_loss={losses['total_loss'].item():.4f}"
          f" intrinsic_mean={losses['intrinsic_reward'].mean().item():.4f}")


def _check_running_mean_std() -> None:
    """Sanity-check RunningMeanStd against a reference torch computation."""
    torch.manual_seed(0)
    data = torch.randn(1000, 4) * 3.0 + 1.5

    rms = RunningMeanStd(shape=(4,))
    for chunk in data.split(50):
        rms.update(chunk)

    ref_mean = data.mean(dim=0)
    ref_var = data.var(dim=0, unbiased=False)
    assert torch.allclose(rms.mean, ref_mean, atol=1e-4), (rms.mean, ref_mean)
    assert torch.allclose(rms.var, ref_var, atol=1e-3), (rms.var, ref_var)

    normalized = rms.normalize(data)
    assert normalized.abs().mean() < 1.0  # standardized data has mean 0
    print("[running-mean-std] OK"
          f" | mean~{rms.mean.mean().item():.3f} std~{rms.std.mean().item():.3f}")


def _check_ppo_trainer() -> None:
    """End-to-end exercise of PPOTrainer on Pendulum-v1.

    Pendulum is bundled with gymnasium core (no MuJoCo install required), is
    continuous-action (1-D), and runs instantly. The config below is
    deliberately tiny -- this verifies wiring, GAE math, and the PPO+ICM
    update path; it is not a learning test.
    """
    try:
        import gymnasium as gym  # noqa: F401
    except ImportError:
        print("[ppo-trainer] SKIP (gymnasium not installed)")
        return

    from drl_curiosity.trainer_ppo import PPOConfig, PPOTrainer

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        return gym.wrappers.RecordEpisodeStatistics(env)

    envs = gym.vector.SyncVectorEnv([make_env, make_env])
    obs_dim = int(envs.single_observation_space.shape[0])
    action_dim = int(envs.single_action_space.shape[0])

    policy = ContinuousActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    icm = ContinuousICM(
        feature_encoder=MLPEncoder(obs_dim=obs_dim),
        action_dim=action_dim,
    )

    config = PPOConfig(
        total_steps=64,
        num_envs=2,
        rollout_steps=16,
        update_epochs=2,
        num_minibatches=4,
        anneal_lr=False,
        log_interval=10_000,  # suppress per-update print
    )

    trainer = PPOTrainer(envs=envs, policy=policy, icm=icm, config=config, logger=None)
    initial_log_std = policy.log_std.detach().clone()
    final_step = trainer.train(seed=0)
    envs.close()

    assert final_step == config.total_steps, (final_step, config.total_steps)
    assert torch.isfinite(policy.log_std).all(), policy.log_std
    # PPO should have moved the log_std at least slightly after 2 updates
    assert not torch.allclose(policy.log_std.detach(), initial_log_std), (
        "log_std did not move -- gradient flow into the policy may be broken"
    )
    print(
        "[ppo-trainer] OK"
        f" | steps={final_step} log_std={policy.log_std.detach().mean().item():+.4f}"
    )


def main() -> None:
    torch.manual_seed(0)
    _check_discrete_pixel_path()
    _check_continuous_state_path()
    _check_running_mean_std()
    _check_ppo_trainer()
    print("smoke test passed")


if __name__ == "__main__":
    main()
