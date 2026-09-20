"""LTX-2.3 SDE index selection matching verl-omni's seeded sparse window."""

from __future__ import annotations

from typing import Optional

import torch

from unirl.sde.index_schedule import TimestepScheduler


class LTX2VerlIndexScheduler(TimestepScheduler):
    """Select three sorted indices from [0, 10) with a rollout-indexed CPU generator."""

    def __init__(self, num_timesteps: int, seed: int = 42) -> None:
        super().__init__(num_timesteps)
        if num_timesteps < 10:
            raise ValueError("LTX2VerlIndexScheduler requires at least 10 denoising steps")
        self.seed = int(seed)

    def get_sde_indices(self, step: Optional[int] = None) -> set[int]:
        generator = torch.Generator().manual_seed(self.seed + (0 if step is None else int(step)))
        return set(torch.randperm(10, generator=generator)[:3].tolist())
