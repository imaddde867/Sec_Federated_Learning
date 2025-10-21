from __future__ import annotations
import math
from typing import Iterable, Tuple, List
import torch

def l2_clip_(grads: Iterable[torch.Tensor], max_norm: float) -> float:
    """
    Clip a list of gradient tensors in-place so their concatenated global L2 norm
    is at most `max_norm`. Returns the *original* global norm (before clipping).
    """
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.view(-1))
    if not flat:
        return 0.0
    vec = torch.cat(flat)
    orig_norm = torch.linalg.vector_norm(vec).item()
    if max_norm > 0 and orig_norm > max_norm:
        scale = max_norm / (orig_norm + 1e-12)
        with torch.no_grad():
            for g in grads:
                if g is not None:
                    g.mul_(scale)
    return orig_norm

def add_gaussian_noise_(grads: Iterable[torch.Tensor], sigma: float, max_norm: float) -> None:
    """
    Add i.i.d. Gaussian noise to each gradient tensor, with std = sigma * max_norm.
    No-op if sigma <= 0.
    """
    if sigma <= 0:
        return
    for g in grads:
        if g is None:
            continue
        noise = torch.normal(
            mean=0.0,
            std=max_norm * sigma,
            size=g.shape,
            device=g.device,
            dtype=g.dtype,
        )
        g.add_(noise)

def dp_sanitize_(grads: Iterable[torch.Tensor], max_norm: float, noise_multiplier: float) -> Tuple[float, float]:
    """
    Apply DP-SGD style clipping + Gaussian noise **in-place**.
    Returns (original_global_norm, noise_multiplier).
    """
    orig = l2_clip_(grads, max_norm)
    add_gaussian_noise_(grads, sigma=noise_multiplier, max_norm=max_norm)
    return orig, noise_multiplier

def rdp_epsilon(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
    orders: List[float] | None = None,
) -> float:
    """
    Lightweight/rough RDP accountant to get an *approximate* ε.
    For rigorous accounting use Opacus or a formal RDP accountant.

    Returns the minimal ε over provided Renyi orders.
    """
    if orders is None:
        orders = [1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    if noise_multiplier <= 0:
        return float("inf")
    q = sample_rate
    eps_candidates = []
    for alpha in orders:
        sigma2 = noise_multiplier * noise_multiplier
        # Very rough upper bound for Poisson subsampled Gaussian
        if sigma2 <= 0.5:
            rho = float("inf")
        else:
            rho = steps * (alpha * q * q) / (2.0 * sigma2 - 1.0)
        eps = rho + math.log(1.0 / max(delta, 1e-12)) / (alpha - 1.0)
        eps_candidates.append(eps)
    return min(eps_candidates)