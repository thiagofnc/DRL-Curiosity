"""HalfCheetah-v4 (or any MuJoCo continuous-control env) entrypoint for the
PPO + ICM trainer.

Usage:
    python scripts/train_halfcheetah.py --total-steps 1000000

This wires together MLPEncoder + ContinuousActorCritic + ContinuousICM and
hands them to drl_curiosity.trainer_ppo.PPOTrainer.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drl_curiosity.encoders import MLPEncoder
from drl_curiosity.icm import ContinuousICM
from drl_curiosity.logging_utils import Logger, make_run_dir
from drl_curiosity.policies import ContinuousActorCritic
from drl_curiosity.trainer_ppo import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO + ICM on a continuous-control env.")
    p.add_argument("--env-id", default="HalfCheetah-v4")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--icm-coef", type=float, default=1.0)
    p.add_argument("--intrinsic-reward-scale", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-normalize-obs", action="store_true")
    p.add_argument("--no-normalize-intrinsic", action="store_true")
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--device", default=None)
    return p.parse_args()


def make_env_fn(env_id: str, seed: int, idx: int):
    def thunk():
        import gymnasium as gym

        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False  # speed > determinism on conv kernels

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Gymnasium is required. Install with `pip install gymnasium mujoco`."
        ) from exc

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env_fn(args.env_id, args.seed, i) for i in range(args.num_envs)]
    )

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    policy = ContinuousActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    icm_encoder = MLPEncoder(obs_dim=obs_dim).to(device)
    icm = ContinuousICM(feature_encoder=icm_encoder, action_dim=action_dim).to(device)

    run_dir = make_run_dir(args.log_dir, args.env_id, "ppo_icm", args.seed)
    logger = Logger(run_dir)
    print(f"logging to {run_dir}")

    config = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        icm_coef=args.icm_coef,
        intrinsic_reward_scale=args.intrinsic_reward_scale,
        max_grad_norm=args.max_grad_norm,
        normalize_obs=not args.no_normalize_obs,
        normalize_intrinsic=not args.no_normalize_intrinsic,
        anneal_lr=not args.no_anneal_lr,
        lr=args.lr,
    )

    trainer = PPOTrainer(
        envs=envs,
        policy=policy,
        icm=icm,
        config=config,
        logger=logger,
        device=device,
    )
    trainer.train(seed=args.seed)

    logger.close()
    envs.close()


if __name__ == "__main__":
    main()
