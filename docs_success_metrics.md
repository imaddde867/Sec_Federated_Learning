# Success Metrics for Federated DP-Encryption Project

## Privacy
- **PSNR (dB)**: Peak Signal-to-Noise Ratio on attacker-reconstructed images vs. ground truth. Higher is *worse for the attacker* (i.e., better privacy). Report mean±std across clients and images.
- **SSIM**: Structural Similarity Index (0–1). Lower is better privacy.
- **LPIPS (optional)**: Learned Perceptual Image Patch Similarity. Higher distance is better privacy.
- **Attack success rate** (optional): fraction of reconstructions exceeding SSIM≥0.5.

## Utility
- **Top‑1 accuracy (%)** on a held-out test set.
- **Convergence**: rounds to reach target accuracy (e.g., 70%) or loss plateau slope.

## Performance / Overhead
- **Time per round (s)**: wall‑clock per FL round (server perspective).
- **Bytes per round (MB)**: total uplink/downlink payload per client and aggregate.

## Reproducibility
- **Seed**: RNG seed controlling data split, init, shuffles.
- **Commit**: Git SHA of code used.
- **Config hash**: SHA256 of normalized config (Hydra dict) -> becomes `exp_id`.
- **Env**: Python version, CUDA/cuDNN, key library versions.