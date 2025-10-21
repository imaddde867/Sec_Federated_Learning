from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Any, Dict, Optional, Iterable
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

@dataclass
class Timer:
    t0: float = 0.0
    dt: float = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0

def latency_throughput(fn: Callable, *args, warmup: int = 1, iters: int = 10, **kwargs) -> Dict[str, float]:
    """
    Run fn(*args, **kwargs) a few times and report mean latency and throughput.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return {
        "latency_ms": (dt / max(iters, 1)) * 1000.0,
        "throughput_ops_s": iters / max(dt, 1e-9),
    }

def compute_psnr(img: np.ndarray, ref: np.ndarray, data_range: Optional[float] = None) -> float:
    if data_range is None:
        data_range = float(np.max(ref) - np.min(ref)) or 1.0
    return float(peak_signal_noise_ratio(ref, img, data_range=data_range))

def compute_ssim(img: np.ndarray, ref: np.ndarray, data_range: Optional[float] = None) -> float:
    if data_range is None:
        data_range = float(np.max(ref) - np.min(ref)) or 1.0
    # If color image (H,W,3), average channel-wise SSIM
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        s = 0.0
        for c in range(img.shape[-1]):
            s += structural_similarity(ref[..., c], img[..., c], data_range=data_range)
        return float(s / img.shape[-1])
    return float(structural_similarity(ref, img, data_range=data_range))

def grad_norms(grads: Iterable[torch.Tensor]) -> Dict[str, float]:
    flat = torch.cat([g.view(-1) for g in grads if g is not None])
    l2 = torch.linalg.vector_norm(flat).item()
    linf = torch.max(torch.abs(flat)).item()
    return {"grad_l2": l2, "grad_linf": linf}