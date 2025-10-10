#!/bin/bash

set -e  # Exit on error

ENV_NAME="fl_privacy"
PYTHON_VERSION="3.9"

# Check if conda is initialized
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

echo "Successfully activated ${ENV_NAME}"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (optimized for Apple Silicon if on macOS)
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon Mac - installing optimized PyTorch"
    pip install torch torchvision
else
    pip install torch torchvision
fi

# Install other requirements
echo "Installing additional packages..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found. Installing packages manually..."
    pip install numpy pandas matplotlib opacus flwr tenseal breaching
fi

echo ""
echo "=== Setup Complete ==="
echo "To activate this environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"