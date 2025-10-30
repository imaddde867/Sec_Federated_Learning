# CAPSTONE
Security in Federated Learning

## Overview
This repository hosts experiments and utilities for benchmarking privacy-preserving techniques in federated learning scenarios. The focus is on image classification with CIFAR-100, selective encryption using TenSEAL, and resilience against model inversion attacks.

## Current Progress
- Extended `src/fl_simulation/fed_avg.ipynb` into a configurable FedAvg run with CIFAR-100, 20 clients sampled 5/round, Dirichlet α=0.5 non-IID splits, and CUDA execution.
- Added telemetry capture per client (gradient/per-layer norms, loss traces, batch sizes, class histograms) plus deterministic seeding and reusable CIFAR-100 loaders.
- Persisted round artifacts via `fedavg_metrics_{round}_meta.json` and companion `_tensors.pt` blobs holding raw gradients, deltas, and server aggregates.
- Completed a 10-round experiment; training diverges from round 4 onward (NaN loss, ~1% accuracy) because the current server delta averaging relies on `Tensor.T` for high-rank tensors, which PyTorch warns will misbehave.

## Next Steps
- Replace the server aggregate calculation with a dimension-safe weighted average (explicit broadcasting instead of `Tensor.T`) and confirm the run no longer hits NaNs.
- Evaluate safeguard options such as gradient clipping or mixed-precision limits before scaling to more rounds or harsher non-IID regimes.
- Script a lightweight report that summarizes the exported JSON/PT metrics across rounds for faster experiment review.
- Once the baseline is stable, resume TenSEAL selective encryption experiments and planned attack simulations with SSIM/PSNR reporting.

## Repository Layout

- `src/` — Reusable Python packages for data handling, model definitions, encryption helpers, attack implementations, and FL simulation utilities.
- `reports/` — Generated metrics, plots, and written summaries.
- `docs/` — Project documentation and planning artefacts.

## Setup Instructions

### Option A (macOS/Linux)
- Ensure Conda/Miniconda is installed and available in your shell.
- From the repo root, run:

```bash
chmod +x setup.sh
./setup.sh
```

If using Git Bash on Windows, you can also run the above `.sh` script from Git Bash.

### Option B (Windows PowerShell / Anaconda Prompt)
- Open PowerShell or Anaconda Prompt where `conda` is available. If needed, initialize once with `conda init powershell` and restart the shell.
- From the repo root, run:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
```

If execution is restricted, run in the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
./setup.ps1
```

### Activate the Environment (all platforms)
Activate the newly created `fl_privacy` conda environment:

```bash
conda activate fl_privacy
```

### Configure VS Code Python Interpreter
To ensure VS Code uses the correct Python environment:

1. Open the Command Palette: `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows/Linux)
2. Type and select "Python: Select Interpreter"
3. Choose the `fl_privacy` environment from the list

This ensures all code runs within the configured environment with the correct dependencies.

## Requirements
- Conda or Miniconda installed
- VS Code with Python extension

## Project Structure
- `data/` - Dataset storage
- `experiments/` - Experimental scripts
- `scripts/` - Utility scripts
- `src/` - Source code

## Run Algorithm
- make sure to have jupyter installed 
```bash
pip install jupyter
```
- run 
```bash
jupyter lab
```
