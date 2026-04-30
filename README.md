# DRL-Curiosity

This project explores curiosity-driven deep reinforcement learning based on the paper **"Curiosity-driven Exploration by Self-supervised Prediction."**

Our first goal is to implement the core approach described in the paper and reproduce its main ideas in code. After that, we plan to add two improvements of our own. These improvements are still undecided and will be chosen as we better understand the method and its limitations.

## Current Implementation

The starter code includes the paper's visual encoder, an A3C-style actor-critic model, and an Intrinsic Curiosity Module with inverse and forward dynamics losses.

Run the smoke test with:

```bash
python scripts/smoke_test.py
```

The first training script is a single-process actor-critic version:

```bash
python train.py --env-id <image-based-gymnasium-env>
```
