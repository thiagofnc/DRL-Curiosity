"""PPO + ICM trainer for continuous-control envs (e.g. MuJoCo HalfCheetah).

Departure from the Pathak et al. 2017 paper: the paper uses A3C; this module
uses PPO because A3C is brittle on MuJoCo and PPO is the de-facto standard
for continuous control. The curiosity machinery (inverse + forward dynamics,
intrinsic reward from forward-prediction error) is the same as the paper --
only the policy-gradient algorithm differs.

Design choices made up front (see project memory for rationale):
- Single combined reward stream `r = r_ext + eta * r_int_normalized` and a
  single value head. RND-style two-head split is a possible future
  improvement but not Phase 2.
- Intrinsic reward is normalized RND-style: divide raw forward errors by a
  running std of discounted intrinsic *returns* (not raw step errors), with
  non-resetting return accumulator. Toggle via `normalize_intrinsic`.
- Observation normalization via RunningMeanStd, clipped to +/-10. Toggle
  via `normalize_obs`.
- Truncation handling: only `terminated` zeroes the bootstrap in GAE.
  `truncated` (time-limit) bootstraps from the next-value estimate.
- One Adam optimizer over policy + ICM parameters together (matches the
  paper's joint loss). Set `icm_coef=0` to disable curiosity for ablations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from drl_curiosity.icm import ContinuousICM
from drl_curiosity.logging_utils import Logger
from drl_curiosity.policies import ContinuousActorCritic
from drl_curiosity.running_stats import RunningMeanStd


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    num_envs: int = 4
    rollout_steps: int = 512  # per env; batch = num_envs * rollout_steps
    update_epochs: int = 10
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    icm_coef: float = 1.0
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    normalize_obs: bool = True
    obs_clip: float = 10.0
    normalize_intrinsic: bool = True
    intrinsic_reward_scale: float = 0.01  # eta, applied AFTER std normalization
    log_interval: int = 1


class RolloutBuffer:
    """Fixed-size per-rollout storage with GAE-lambda advantage computation.

    Layout: tensors of shape (T, N, *) where T = rollout_steps, N = num_envs.
    Stores the normalized obs (for the policy / ICM) plus per-step actions,
    log-probs, values, the *combined* (extrinsic + scaled intrinsic) reward,
    and termination flags. `truncations` is intentionally not stored: GAE
    cares only about whether the next-step bootstrap is valid, and we treat
    truncation as non-terminal.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        T, N = rollout_steps, num_envs
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.next_obs = torch.zeros(T, N, obs_dim, device=device)
        self.actions = torch.zeros(T, N, action_dim, device=device)
        self.log_probs = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, N, device=device)
        self.rewards = torch.zeros(T, N, device=device)
        self.terminations = torch.zeros(T, N, device=device)
        self.T = T
        self.N = N
        self.device = device

    def store(
        self,
        t: int,
        obs: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        termination: Tensor,
        next_obs: Tensor,
    ) -> None:
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.terminations[t] = termination
        self.next_obs[t] = next_obs

    def compute_gae(
        self,
        last_value: Tensor,
        last_termination: Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_nonterminal = 1.0 - last_termination
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.terminations[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns

    def flatten(self) -> dict[str, Tensor]:
        return {
            "obs": self.obs.reshape(self.T * self.N, -1),
            "next_obs": self.next_obs.reshape(self.T * self.N, -1),
            "actions": self.actions.reshape(self.T * self.N, -1),
            "log_probs": self.log_probs.reshape(-1),
            "values": self.values.reshape(-1),
        }


def _extract_episode_stats(infos: Any) -> tuple[list[float], list[int]]:
    """Pull completed-episode returns/lengths out of a gymnasium vector env
    info dict. Tolerant to both the 0.29-style (`infos["episode"]` is a dict
    of per-env arrays with a mask in `infos["_episode"]`) and the looser
    list-of-dicts style.
    """
    returns: list[float] = []
    lengths: list[int] = []
    if not isinstance(infos, dict):
        return returns, lengths
    ep_info = infos.get("episode")
    if ep_info is None:
        return returns, lengths
    if isinstance(ep_info, dict) and "r" in ep_info:
        mask = infos.get("_episode")
        rs = np.asarray(ep_info["r"]).reshape(-1)
        ls = np.asarray(ep_info["l"]).reshape(-1)
        if mask is not None:
            mask = np.asarray(mask).reshape(-1).astype(bool)
            for i, m in enumerate(mask):
                if m:
                    returns.append(float(rs[i]))
                    lengths.append(int(ls[i]))
        else:
            returns.extend(float(r) for r in rs)
            lengths.extend(int(l) for l in ls)
    return returns, lengths


class PPOTrainer:
    """Single-file PPO+ICM trainer wired against a gymnasium vector env."""

    def __init__(
        self,
        envs: Any,
        policy: ContinuousActorCritic,
        icm: ContinuousICM,
        config: PPOConfig,
        logger: Logger | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.envs = envs
        self.policy = policy
        self.icm = icm
        self.config = config
        self.logger = logger
        self.device = device or torch.device("cpu")

        single_obs_space = envs.single_observation_space
        single_act_space = envs.single_action_space
        self.obs_dim = int(np.prod(single_obs_space.shape))
        self.action_dim = int(np.prod(single_act_space.shape))

        self.action_low = torch.as_tensor(
            single_act_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            single_act_space.high, dtype=torch.float32, device=self.device
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.icm.parameters()),
            lr=config.lr,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=config.num_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.obs_rms = (
            RunningMeanStd(shape=(self.obs_dim,)) if config.normalize_obs else None
        )
        self.intrinsic_rms = (
            RunningMeanStd(shape=()) if config.normalize_intrinsic else None
        )
        # Non-resetting per-env discounted intrinsic return (RND recipe).
        self.intrinsic_running_return = torch.zeros(config.num_envs)

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs).clamp(
            -self.config.obs_clip, self.config.obs_clip
        )

    def _scale_intrinsic_reward(self, raw_forward_error: Tensor) -> Tensor:
        """RND-style normalization: track running std of discounted intrinsic
        returns, divide raw forward errors by that std, then apply eta.
        """
        if not self.config.normalize_intrinsic or self.intrinsic_rms is None:
            return self.config.intrinsic_reward_scale * raw_forward_error
        raw_cpu = raw_forward_error.detach().cpu()
        self.intrinsic_running_return = (
            self.intrinsic_running_return * self.config.gamma + raw_cpu
        )
        self.intrinsic_rms.update(self.intrinsic_running_return)
        std = float(self.intrinsic_rms.std)
        return self.config.intrinsic_reward_scale * (raw_forward_error / (std + 1e-8))

    def collect_rollout(self, raw_obs_np: np.ndarray) -> dict[str, Any]:
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        intrinsic_raw_sum = 0.0
        intrinsic_scaled_sum = 0.0
        last_termination = torch.zeros(self.config.num_envs, device=self.device)

        for t in range(self.config.rollout_steps):
            raw_obs = torch.as_tensor(raw_obs_np, dtype=torch.float32, device=self.device)
            if self.obs_rms is not None:
                self.obs_rms.update(raw_obs.cpu())
            obs_normalized = self._normalize_obs(raw_obs)

            with torch.no_grad():
                dist, value = self.policy.distribution(obs_normalized)
                action_sampled = dist.sample()
                log_prob = dist.log_prob(action_sampled).sum(-1)

            action_clipped = torch.maximum(
                torch.minimum(action_sampled, self.action_high), self.action_low
            )
            next_obs_np, reward_np, terminated, truncated, infos = self.envs.step(
                action_clipped.cpu().numpy()
            )

            raw_next_obs = torch.as_tensor(
                next_obs_np, dtype=torch.float32, device=self.device
            )
            if self.obs_rms is not None:
                # Note: do NOT update RMS with next_obs again -- we already updated
                # with raw_obs at the head of this step; updating twice biases stats.
                pass
            next_obs_normalized = self._normalize_obs(raw_next_obs)

            with torch.no_grad():
                icm_out = self.icm.losses(
                    obs_normalized, next_obs_normalized, action_sampled
                )
            raw_forward_error = icm_out["forward_error"]
            intrinsic_reward = self._scale_intrinsic_reward(raw_forward_error)

            reward_ext = torch.as_tensor(
                reward_np, dtype=torch.float32, device=self.device
            )
            total_reward = reward_ext + intrinsic_reward
            termination_t = torch.as_tensor(
                terminated, dtype=torch.float32, device=self.device
            )

            self.buffer.store(
                t,
                obs=obs_normalized,
                action=action_sampled,
                log_prob=log_prob,
                value=value,
                reward=total_reward,
                termination=termination_t,
                next_obs=next_obs_normalized,
            )

            intrinsic_raw_sum += float(raw_forward_error.mean().item())
            intrinsic_scaled_sum += float(intrinsic_reward.mean().item())

            returns_seen, lengths_seen = _extract_episode_stats(infos)
            ep_returns.extend(returns_seen)
            ep_lengths.extend(lengths_seen)

            raw_obs_np = next_obs_np
            last_termination = termination_t

        stats: dict[str, float] = {
            "intrinsic_raw_mean": intrinsic_raw_sum / self.config.rollout_steps,
            "intrinsic_scaled_mean": intrinsic_scaled_sum / self.config.rollout_steps,
        }
        if ep_returns:
            stats["episode_return_mean"] = float(np.mean(ep_returns))
            stats["episode_return_std"] = float(np.std(ep_returns))
            stats["episode_length_mean"] = float(np.mean(ep_lengths))
            stats["episodes_completed"] = float(len(ep_returns))
        return {
            "last_raw_obs": raw_obs_np,
            "last_termination": last_termination,
            "stats": stats,
        }

    def update(
        self, last_raw_obs_np: np.ndarray, last_termination: Tensor
    ) -> dict[str, float]:
        last_raw = torch.as_tensor(
            last_raw_obs_np, dtype=torch.float32, device=self.device
        )
        last_obs_normalized = self._normalize_obs(last_raw)
        with torch.no_grad():
            _, _, last_value = self.policy(last_obs_normalized)

        advantages, returns = self.buffer.compute_gae(
            last_value,
            last_termination,
            self.config.gamma,
            self.config.gae_lambda,
        )

        flat = self.buffer.flatten()
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        batch_size = self.buffer.T * self.buffer.N
        mb_size = max(1, batch_size // self.config.num_minibatches)
        indices = np.arange(batch_size)

        sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "icm_loss": 0.0,
            "icm_inverse_loss": 0.0,
            "icm_forward_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
        }
        n_updates = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_idx = torch.as_tensor(
                    indices[start:end], dtype=torch.long, device=self.device
                )

                mb_obs = flat["obs"][mb_idx]
                mb_next_obs = flat["next_obs"][mb_idx]
                mb_actions = flat["actions"][mb_idx]
                mb_old_log_probs = flat["log_probs"][mb_idx]
                mb_old_values = flat["values"][mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_ret = ret_flat[mb_idx]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                dist, new_values = self.policy.distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                    )

                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.config.clip_value_loss:
                    v_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                        -self.config.clip_coef, self.config.clip_coef
                    )
                    vf_losses1 = (new_values - mb_ret).pow(2)
                    vf_losses2 = (v_clipped - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_ret).pow(2).mean()

                icm_losses = self.icm.losses(mb_obs, mb_next_obs, mb_actions)
                icm_total = icm_losses["total_loss"]

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy
                    + self.config.icm_coef * icm_total
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.icm.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                sums["policy_loss"] += float(policy_loss.item())
                sums["value_loss"] += float(value_loss.item())
                sums["entropy"] += float(entropy.item())
                sums["icm_loss"] += float(icm_total.item())
                sums["icm_inverse_loss"] += float(icm_losses["inverse_loss"].item())
                sums["icm_forward_loss"] += float(icm_losses["forward_loss"].item())
                sums["approx_kl"] += float(approx_kl.item())
                sums["clip_frac"] += float(clip_frac.item())
                sums["grad_norm"] += float(grad_norm)
                n_updates += 1

        return {k: v / max(1, n_updates) for k, v in sums.items()}

    def train(self, seed: int = 0) -> int:
        raw_obs_np, _ = self.envs.reset(seed=seed)
        global_step = 0
        steps_per_update = self.config.num_envs * self.config.rollout_steps
        num_updates = max(1, self.config.total_steps // steps_per_update)
        start_time = time.time()
        config_dict = {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}
        if self.logger is not None:
            self.logger.dump_config(config_dict)

        for update in range(1, num_updates + 1):
            if self.config.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.lr * frac

            rollout = self.collect_rollout(raw_obs_np)
            raw_obs_np = rollout["last_raw_obs"]
            global_step += steps_per_update

            update_info = self.update(raw_obs_np, rollout["last_termination"])

            sps = int(global_step / max(1e-6, time.time() - start_time))
            if self.logger is not None:
                self.logger.scalar("charts/SPS", sps, global_step)
                self.logger.scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                self.logger.scalar(
                    "charts/policy_log_std_mean",
                    float(self.policy.log_std.mean().item()),
                    global_step,
                )
                for k, v in update_info.items():
                    self.logger.scalar(f"losses/{k}", v, global_step)
                for k, v in rollout["stats"].items():
                    self.logger.scalar(f"rollout/{k}", v, global_step)

            if update % self.config.log_interval == 0:
                ep_ret = rollout["stats"].get("episode_return_mean", float("nan"))
                print(
                    f"step={global_step:>8d} update={update:>4d} SPS={sps:>5d} "
                    f"policy={update_info['policy_loss']:+.4f} "
                    f"value={update_info['value_loss']:.4f} "
                    f"icm={update_info['icm_loss']:.4f} "
                    f"kl={update_info['approx_kl']:.4f} "
                    f"ep_ret={ep_ret:.2f}"
                )

        return global_step
