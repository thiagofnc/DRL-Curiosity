from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def preprocess_frame(frame: np.ndarray, size: int = 42) -> Tensor:
    """Convert an RGB or grayscale frame to a normalized 42x42 grayscale tensor."""

    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    frame_tensor = torch.as_tensor(frame, dtype=torch.float32)
    if frame_tensor.max() > 1.0:
        frame_tensor = frame_tensor / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
    frame_tensor = F.interpolate(frame_tensor, size=(size, size), mode="bilinear", align_corners=False)
    return frame_tensor.squeeze(0)


def stack_frames(frames: list[Tensor], stack_size: int = 4) -> Tensor:
    """Create the paper's state representation from the most recent grayscale frames."""

    if not frames:
        raise ValueError("frames must contain at least one frame")
    padded = [frames[0]] * max(0, stack_size - len(frames)) + frames[-stack_size:]
    return torch.cat(padded, dim=0)
