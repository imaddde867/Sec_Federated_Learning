"""Verify the FL Privacy setup"""

import sys

def check_imports():
    """Check if all required packages can be imported"""
    packages = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'opacus': 'Opacus',
        'flwr': 'Flower',
        'tenseal': 'TenSEAL',
        'breaching': 'Breaching',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }
    
    print("Checking package imports...")
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0

def check_torch():
    """Check PyTorch configuration"""
    import torch
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU acceleration available")
    elif torch.cuda.is_available():
        print(f"✓ CUDA GPU acceleration available (device: {torch.cuda.get_device_name(0)})")
    else:
        print("ℹ Running on CPU only")

def main():
    print("FL Privacy Environment Verification")
    
    success = check_imports()
    print()
    check_torch()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Setup verification PASSED")
        print("=" * 50)
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
