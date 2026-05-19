"""Load a trained PPO+ICM checkpoint and record MP4 videos of the policy.

Usage:
    python scripts/play_halfcheetah.py runs/HalfCheetah-v5_ppo_icm_s0_<...>
    python scripts/play_halfcheetah.py path/to/checkpoint.pt --episodes 5

You can pass either the run directory (we look for `checkpoint.pt` inside)
or the checkpoint file path directly. Videos land in `<run_dir>/videos/` by
default, or `--video-dir <path>` to override.

The policy is evaluated *deterministically* (Gaussian mean, not a sample) --
that's the right thing for a "watch the trained agent" eval, not training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drl_curiosity.policies import ContinuousActorCritic
from drl_curiosity.running_stats import RunningMeanStd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record MP4 videos of a trained PPO+ICM agent."
    )
    p.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint.pt OR to the run directory containing it.",
    )
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video-dir",
        default=None,
        help="Where to write MP4s. Defaults to <checkpoint_dir>/videos.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the Gaussian mean.",
    )
    return p.parse_args()


def _resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    return path


def _restore_obs_rms(ckpt_obs_rms: dict | None, obs_dim: int) -> RunningMeanStd | None:
    if ckpt_obs_rms is None:
        return None
    rms = RunningMeanStd(shape=(obs_dim,))
    rms.mean = ckpt_obs_rms["mean"]
    rms.var = ckpt_obs_rms["var"]
    rms.count = ckpt_obs_rms["count"]
    return rms


def main() -> None:
    args = parse_args()

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit("Gymnasium is required. `pip install gymnasium mujoco imageio`.") from exc

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    video_dir = (
        Path(args.video_dir) if args.video_dir else ckpt_path.parent / "videos"
    )
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
    name_prefix = f"halfcheetah_step{int(ckpt.get('global_step', 0))}"
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda _: True,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    policy = ContinuousActorCritic(
        obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"]
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    obs_rms = _restore_obs_rms(ckpt.get("obs_rms"), ckpt["obs_dim"])

    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    print(f"playing {args.episodes} episode(s) of {args.env_id}")
    print(f"checkpoint: {ckpt_path} (trained for {int(ckpt.get('global_step', 0))} steps)")
    print(f"videos -> {video_dir}")
    print(f"mode: {'stochastic' if args.stochastic else 'deterministic (Gaussian mean)'}")

    returns: list[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if obs_rms is not None:
                obs_t = obs_rms.normalize(obs_t).clamp(-10.0, 10.0)
            with torch.no_grad():
                if args.stochastic:
                    dist, _ = policy.distribution(obs_t)
                    action_t = dist.sample()
                else:
                    mu, _, _ = policy(obs_t)
                    action_t = mu
            action = (
                torch.maximum(torch.minimum(action_t, action_high), action_low)
                .squeeze(0)
                .numpy()
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
        returns.append(total_reward)
        print(f"  episode {ep + 1}: return={total_reward:.2f} length={steps}")

    env.close()
    print(f"mean return over {args.episodes} ep(s): {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"videos saved under {video_dir}")


if __name__ == "__main__":
    main()
