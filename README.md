# CAPSTONE
Security in Federated Learning

## Overview
This repository hosts experiments and utilities for benchmarking privacy-preserving techniques in federated learning scenarios. The focus is on image classification with CIFAR-100, selective encryption using TenSEAL, and resilience against model inversion attacks.

## Current Progress
- Implemented an end-to-end FedAvg prototype in `src/fl_simulation/fed_avg.ipynb` using ResNet18 on CIFAR-100 with three IID clients and five communication rounds, configured for Apple `mps` acceleration.
- Set up IID partitioning, local SGD training with momentum, and global aggregation/evaluation, reaching ~41.9% top-1 accuracy on the CIFAR-100 test set after five rounds.
- Automated dataset acquisition via `torchvision` and validated the training loop across all rounds.

## Next Steps
- Turn the notebook prototype into a configurable FL harness (parameterized clients/rounds/devices, tracked round metrics, persisted artifacts) in line with `docs/tasks_overview.md`.
- Expand data handling to support alternative architectures (VGG16, custom CNNs) and non-IID sampling, preparing for attack/evaluation scenarios.
- Integrate TenSEAL-based selective encryption for client updates and stage Yin/Random/Confusion attack simulations with PSNR/SSIM reporting.
- Add regression checks or lightweight CI scripts to protect the baseline as security and attack modules evolve.

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