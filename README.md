# DRL-Curiosity

Curiosity-driven deep RL based on **"Curiosity-driven Exploration by Self-supervised Prediction"** ([Pathak et al., 2017](./curiosity_driven_exploration_paper.pdf)).

## Goal

Reproduce the paper's Intrinsic Curiosity Module (ICM), then extend it to MuJoCo HalfCheetah and add two improvements of our own (still TBD).

## Implementation

Reusable components under `drl_curiosity/`:

- `encoders.py` — `ConvEncoder` (paper's 4-layer ELU conv on 42×42 stacked frames) + `MLPEncoder` (state vectors).
- `policies.py` — `DiscreteActorCritic` (conv + LSTM + Categorical) + `ContinuousActorCritic` (MLP + diagonal Gaussian, learnable `log_std`).
- `icm.py` — `DiscreteICM` (paper-faithful) + `ContinuousICM` (MSE inverse, raw-action forward). Encoder is injected so ICM keeps its own, as in the paper.
- `running_stats.py` — Welford `RunningMeanStd` for observation / intrinsic-reward normalization.
- `logging_utils.py` — thin TensorBoard wrapper.
- `trainer_ppo.py` — PPO + ICM trainer: vectorized envs, GAE-λ, clipped policy + value losses, joint Adam over policy + ICM, linear LR anneal, end-of-training checkpoint.

Entrypoints under `scripts/`:

- `train_halfcheetah.py` — train PPO+ICM on `HalfCheetah-v5` (or any continuous-action gymnasium env).
- `play_halfcheetah.py` — load a checkpoint and record MP4s of the trained agent.
- `smoke_test.py` — verifies discrete/pixel, continuous/state, `RunningMeanStd`, and PPO end-to-end (no MuJoCo install required).

`train.py` is the original single-process A3C-discrete trainer for the paper's pixel pipeline. Multi-worker A3C is upcoming.

## Usage

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/train_halfcheetah.py --total-steps 1000000
python scripts/play_halfcheetah.py runs/<run_dir>
tensorboard --logdir runs
```

## ICM on HalfCheetah — deviations from the paper

The paper applies ICM with A3C on pixel-based discrete-action envs (VizDoom, Mario). HalfCheetah is state-based and continuous-action, so a few pieces had to change:

- **Inverse model: MSE regression on action vectors**, not cross-entropy over action classes.
- **Forward model: raw action vector input**, not one-hot.
- **PPO instead of A3C.** A3C is brittle on MuJoCo; PPO is the de-facto continuous-control algorithm. The curiosity machinery is identical — only the policy-gradient algorithm differs.
- **MLP encoder** in place of the conv encoder (17-D state vector, not pixels).
- **Running-std intrinsic-reward normalization** (RND-style): divide raw forward errors by a running estimate of the std of *discounted intrinsic returns*, then apply η. Without this, intrinsic-reward scale drifts as feature representations evolve. Disable with `--no-normalize-intrinsic` for a paper-faithful ablation.
- **Observation normalization** (`RunningMeanStd`, clipped to ±10) — essentially required for PPO to learn on MuJoCo. Disable with `--no-normalize-obs`.
- **Single combined reward stream:** `r = r_ext + η · r_int_normalized` with one value head, matching the paper's joint formulation. An RND-style two-head split (separate value functions for extrinsic / intrinsic with different γ) is a candidate future improvement.

HalfCheetah has **dense rewards**, so curiosity is not expected to help much here — this is a wiring test. The interesting comparisons will come from sparse-reward variants.

## Status

- ✅ Phase 1 — refactor backbone into reusable components.
- ✅ Phase 2 — PPO + continuous ICM, HalfCheetah training, MP4 playback.
- ⏳ Phase 3 — multi-worker A3C, sparse-reward ablations, eval script.
- ⏳ Phase 4 — two improvements / contributions (TBD).
