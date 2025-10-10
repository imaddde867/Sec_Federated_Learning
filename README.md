# CAPSTONE
Security in Federated Learning

## Overview
This project focuses on security and privacy aspects of federated learning systems.

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