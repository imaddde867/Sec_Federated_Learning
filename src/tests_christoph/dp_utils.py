"""
Minimal DP utilities (placeholders) to keep the FL runner clean.
- clip_gradients_inplace: per-parameter gradient clipping to max L2 norm.
- add_noise_to_update: add i.i.d. Gaussian noise to each tensor in an update dict.

Note: These are simple hooks, not a full DP accountant. Integrate Opacus or
custom accounting later without changing FLRunner.
"""
from __future__ import annotations
from typing import Dict, Iterable

import torch


def clip_gradients_inplace(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    if max_norm is None or max_norm <= 0:
        return
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(p.grad.data.norm(2).item() ** 2)
    total_norm = total_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)


def add_noise_to_update(delta: Dict[str, torch.Tensor], sigma: float) -> Dict[str, torch.Tensor]:
    if sigma is None or sigma <= 0:
        return delta
    noisy = {}
    for k, v in delta.items():
        noise = torch.randn_like(v) * float(sigma)
        noisy[k] = v + noise
    return noisy
