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

### Step 1: Run the Setup Script
First, make the setup script executable and run it to create the conda environment:

```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Activate the Environment
Activate the newly created `fl_privacy` conda environment:

```bash
conda activate fl_privacy
```

### Step 3: Configure VS Code Python Interpreter
To ensure VS Code uses the correct Python environment:

1. Press `Cmd + Shift + P` (macOS) to open the Command Palette
2. Type and select **"Python: Select Interpreter"**
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
