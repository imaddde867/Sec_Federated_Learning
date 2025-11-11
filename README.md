# Security in Federated Learning

## Overview
This repository contains a compact, configurable Federated Learning (FL) runner with optional homomorphic encryption (HE) and differential privacy (DP) hooks. The current focus is CIFAR‑100 with ResNet‑18, selective layer encryption using TenSEAL (CKKS/BFV) or a fast mock adapter, plus utilities to export per‑client updates for downstream attack analysis.

## What’s Included (core-safe)
- `src/core-safe/fl_env.py`: FLConfig + FLRunner
  - CIFAR‑100 loaders, Dirichlet non‑IID partitioning, ResNet‑18, FedAvg.
  - Optional DP hooks: in‑place gradient clipping and Gaussian noise on updates.
  - Optional HE path: encrypt client updates and homomorphically aggregate.
- `src/core-safe/encryption_adapter.py`: HE adapters
  - Mock adapter (no real crypto) for fast testing.
  - TenSEAL CKKS/BFV adapters with chunking for large tensors and weighted aggregation; automatic fallback to mock if TenSEAL isn’t available.
  - SelectiveEncryptionAdapter to encrypt specific layers, gradients, updates, or client‑server comms; exposes simple `encrypt_update` and `decrypt_and_aggregate` APIs and metrics.
- `src/core-safe/dp_utils.py`: Minimal DP utilities (clip grads, add Gaussian noise). Not a full accountant; integrate Opacus later if needed.
- `src/core-safe/run_experiment.py`: CLI to run FL experiments and emit metrics JSON.
- `src/core-safe/enc_test.ipynb`: Notebook baseline for FedAvg with encryption configuration and telemetry capture.
- `src/core-safe/selective_encrypt.py`: Placeholder note for selective encryption experiments.

Attack utilities live under `attacks/` and can consume exported client updates.

## Quick Start
1) Environment
- Use the provided scripts to create a Conda env with PyTorch and torchvision.

macOS/Linux
```bash
chmod +x setup.sh
./setup.sh
conda activate fl_privacy
```

Windows (PowerShell)
```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
conda activate fl_privacy
```

2) Optional: TenSEAL for real HE
- TenSEAL is optional. If not installed, the system falls back to the mock adapter.
- To use CKKS/BFV, install TenSEAL per its docs, then select `--method ckks` or `--method bfv`.

3) Run an experiment (CLI)
```bash
python src/core-safe/run_experiment.py \
  --rounds 1 --clients 10 --cpr 5 --epochs 1 --batch 32 --alpha 0.1 \
  --enc --method mock --layers fc --out reports/example_metrics.json
```
- Flags:
  - `--enc`: enable encryption; `--method mock|ckks|bfv` selects adapter.
  - `--layers`: limit encryption to layers whose names match entries (e.g., `fc`).
  - `--dp`, `--clip`, `--noise`: enable DP hooks and set clip norm and noise scale.
  - `--out`: save aggregated metrics as JSON; CIFAR‑100 downloads to `./data` on first run.

4) From notebooks
- Jupyter is supported; see `src/core-safe/enc_test.ipynb` for a configured FedAvg baseline with telemetry and encryption options.

## Artifacts and Metrics
- Metrics include per‑round timing, test accuracy/loss, and HE adapter stats (`encryption_time`, `decryption_time`, bytes processed).
- Set `export_client_updates=True` in `FLConfig` to persist plaintext client deltas per round under `reports/<experiment>/client_updates/` for analysis and attack simulations.

## Notes and Status
- HE: CKKS/BFV adapters support automatic chunking and weighted homomorphic aggregation, with a safe fallback to a mock adapter when TenSEAL is unavailable.
- DP: hooks are minimal (clip + Gaussian noise) and do not include privacy accounting yet.
- Datasets/Model: defaults to CIFAR‑100 + ResNet‑18; GPU is used if available.

## Repository Pointers
- `src/README.md` explains the workflow: keep stable code in `core-safe/`, experiment in personal subfolders, then upstream via PR.
- `docs/` holds planning notes; `attacks/` contains scripts/utilities for evaluating leakage.

## Requirements
- Conda/Miniconda, Python 3.10+
- PyTorch + torchvision (installed by setup scripts)
- Optional: TenSEAL for CKKS/BFV
