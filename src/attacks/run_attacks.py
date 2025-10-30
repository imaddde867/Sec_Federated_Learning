"""
Main Attack Script - Works with your team's FL metrics
Save this as: src/attacks/run_attacks.py

Usage:
    python src/attacks/run_attacks.py
"""

import pickle
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

# Force CPU for all operations (avoid MPS device issues on Windows)
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import the metrics calculator from the file we just created
from quality_metrics import ImageQualityMetrics


class SimpleGradientAttack:
    """Simple gradient inversion attack."""
    
    def __init__(self, model, max_iterations=500):
        self.model = model
        self.max_iterations = max_iterations
        self.device = next(model.parameters()).device
    
    def attack(self, gradients, num_images=1):
        """Try to reconstruct images from gradients."""
        print(f"  🎯 Running attack ({self.max_iterations} iterations)...")
        
        # Start with random noise
        reconstructed = torch.randn(
            num_images, 3, 32, 32,
            requires_grad=True,
            device=self.device
        )
        
        optimizer = torch.optim.Adam([reconstructed], lr=0.1)
        
        # Convert gradients dict to list
        target_grads = [gradients[name].to(self.device) for name in sorted(gradients.keys())]
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(reconstructed)
            dummy_labels = torch.zeros(num_images, dtype=torch.long, device=self.device)
            loss = torch.nn.functional.cross_entropy(outputs, dummy_labels)
            loss.backward()
            
            # Compare gradients
            current_grads = [p.grad.clone() for p in self.model.parameters() if p.grad is not None]
            grad_diff = sum(((curr - target) ** 2).sum() for curr, target in zip(current_grads, target_grads))
            
            # Optimize
            grad_diff.backward()
            optimizer.step()
            
            with torch.no_grad():
                reconstructed.clamp_(0, 1)
            
            if (iteration + 1) % 100 == 0:
                print(f"    Iteration {iteration+1}/{self.max_iterations}, Loss: {grad_diff.item():.2e}")
        
        return reconstructed.detach()


def move_to_cpu(obj):
    """Recursively move all tensors to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    return obj


def main():
    print("=" * 70)
    print("FEDERATED LEARNING ATTACK PIPELINE")
    print("=" * 70)
    
    # Step 1: Load the metrics file
    metrics_path = Path("src/fl_simulation/fed_metrics_round20.pkl")
    print(f"\n📂 Loading metrics from: {metrics_path}")
    
    if not metrics_path.exists():
        print(f"❌ ERROR: File not found!")
        print(f"   Looking for: {metrics_path.absolute()}")
        print(f"\n💡 Make sure you're running from the CAPSTONE root directory:")
        print(f"   cd /path/to/CAPSTONE")
        print(f"   python src/attacks/run_attacks.py")
        return
    
    # Load with pickle and move all tensors to CPU
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    # Move all tensors from MPS to CPU
    metrics = move_to_cpu(metrics)
    
    print(f"✅ Loaded metrics!")
    print(f"   Round: {metrics['round']}")
    print(f"   Global Accuracy: {metrics['global_accuracy']:.2%}")
    print(f"   Participating Clients: {metrics['participating_clients']}")
    
    # Step 2: Load the model
    print(f"\n🤖 Loading ResNet18 model...")
    model = models.resnet18(num_classes=100)
    model.load_state_dict(metrics['global_model_state'])
    model.eval()
    print(f"✅ Model loaded!")
    
    # Step 3: Load CIFAR-100 for ground truth (we need this to calculate metrics)
    print(f"\n📊 Loading CIFAR-100 test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    print(f"✅ Dataset loaded: {len(test_dataset)} images")
    
    # Step 4: Get a ground truth image (for comparison)
    def get_test_image():
        img, _ = test_dataset[0]
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        img = img * std + mean
        return torch.clamp(img, 0, 1).unsqueeze(0)
    
    ground_truth = get_test_image()
    
    # Step 5: Attack first client as a test
    client_id = metrics['participating_clients'][0]
    print(f"\n🎯 ATTACKING CLIENT {client_id}")
    print(f"   Samples: {metrics['client_metrics'][client_id]['num_samples']}")
    print(f"   Gradient Norm: {metrics['client_metrics'][client_id]['gradient_norm']:.2f}")
    
    gradients = metrics['raw_gradients'][client_id]
    print(f"   Gradient tensors: {len(gradients)}")
    
    # Step 6: Run the attack
    attacker = SimpleGradientAttack(model)
    reconstructed = attacker.attack(gradients, num_images=1)
    
    # Step 7: Calculate metrics
    print(f"\n📊 Calculating quality metrics...")
    evaluator = ImageQualityMetrics()
    results = evaluator.evaluate(ground_truth, reconstructed)
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"PSNR:  {results[0]['psnr']:.2f} dB")
    print(f"SSIM:  {results[0]['ssim']:.4f}")
    if results[0]['lpips'] is not None:
        print(f"LPIPS: {results[0]['lpips']:.4f}")
    
    print(f"\n💡 INTERPRETATION:")
    if results[0]['psnr'] > 25:
        print(f"   ⚠️  HIGH RISK: Reconstruction is quite good (privacy compromised)")
    elif results[0]['psnr'] > 20:
        print(f"   ⚠️  MODERATE: Some features are visible")
    else:
        print(f"   ✅ GOOD: Reconstruction is poor (privacy preserved)")
    
    # Step 8: Visualize
    print(f"\n🎨 Creating visualization...")
    output_dir = Path("reports/attacks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original
    orig_img = ground_truth[0].cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(orig_img)
    axes[0].set_title("Ground Truth Image")
    axes[0].axis('off')
    
    # Reconstructed
    recon_img = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(recon_img)
    axes[1].set_title(f"Reconstructed by Attacker\nPSNR: {results[0]['psnr']:.1f}dB, SSIM: {results[0]['ssim']:.3f}")
    axes[1].axis('off')
    
    plt.suptitle(f"Gradient Inversion Attack - Client {client_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f"attack_client_{client_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {save_path}")
    
    # Step 9: Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'client_id': client_id,
        'metrics': results[0],
        'client_info': metrics['client_metrics'][client_id]
    }
    
    results_path = output_dir / f"results_client_{client_id}.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✅ Saved results to: {results_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ ATTACK COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Check the visualization: {save_path}")
    print(f"2. Read the results: {results_path}")
    print(f"3. Try attacking other clients (modify client_id in the code)")
    print(f"4. Move on to Task A3: Confusion defense testing")


if __name__ == "__main__":
    main()