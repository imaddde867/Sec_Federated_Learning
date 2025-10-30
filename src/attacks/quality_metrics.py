import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("⚠️  LPIPS not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False


class ImageQualityMetrics:
    """Calculate PSNR, SSIM, and LPIPS for reconstructed images."""
    
    def __init__(self, device='cpu'):
        self.device = device
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
    
    def calculate_psnr(self, original, reconstructed):
        """Calculate PSNR. Higher = worse privacy (better reconstruction)."""
        return psnr(original, reconstructed, data_range=1.0)
    
    def calculate_ssim(self, original, reconstructed):
        """Calculate SSIM. Higher = worse privacy (better reconstruction)."""
        if original.ndim == 3:
            return ssim(original, reconstructed, data_range=1.0, 
                       channel_axis=2, multichannel=True)
        return ssim(original, reconstructed, data_range=1.0)
    
    def calculate_lpips(self, original, reconstructed):
        """Calculate LPIPS. Lower = worse privacy (better reconstruction)."""
        if not LPIPS_AVAILABLE:
            return None
        
        def to_lpips_format(img):
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = img * 2.0 - 1.0
            return img.to(self.device)
        
        with torch.no_grad():
            orig_tensor = to_lpips_format(original)
            recon_tensor = to_lpips_format(reconstructed)
            distance = self.lpips_model(orig_tensor, recon_tensor)
        
        return distance.item()
    
    def evaluate(self, original, reconstructed):
        """Calculate all metrics. Returns dict with psnr, ssim, lpips."""
        results = []
        for i in range(original.shape[0]):
            orig_np = original[i].cpu().numpy().transpose(1, 2, 0)
            recon_np = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
            
            results.append({
                'psnr': float(self.calculate_psnr(orig_np, recon_np)),
                'ssim': float(self.calculate_ssim(orig_np, recon_np)),
                'lpips': self.calculate_lpips(orig_np, recon_np)
            })
        
        return results