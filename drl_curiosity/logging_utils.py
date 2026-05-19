from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Logger:
    """Thin TensorBoard wrapper + run-dir bookkeeping.

    SummaryWriter is imported lazily so importing this module does not pull
    tensorboard in (handy for the smoke test on machines without it).
    """

    def __init__(self, run_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def scalars(self, tag_to_value: dict[str, float], step: int) -> None:
        for tag, val in tag_to_value.items():
            self.writer.add_scalar(tag, float(val), step)

    def dump_config(self, config: dict[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True, default=str)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def make_run_dir(base: str | Path, env_id: str, algo: str, seed: int) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_env = str(env_id).replace("/", "_").replace("\\", "_")
    return Path(base) / f"{safe_env}_{algo}_s{seed}_{stamp}"
