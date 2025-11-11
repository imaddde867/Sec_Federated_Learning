"""
Gradient Quality Metrics for Federated Learning Privacy Attacks
Save as: src/attacks/quality_metrics.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path

class GradientQualityMetrics:
    """Calculate similarity metrics between original and reconstructed gradients."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def cosine_similarity(self, grad1, grad2):
        """Calculate cosine similarity between two gradients. Higher = more similar."""
        # Flatten gradients to 1D vectors
        flat1 = torch.cat([g.flatten() for g in grad1.values()])
        flat2 = torch.cat([g.flatten() for g in grad2.values()])
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(flat1.unsqueeze(0), 
                                     flat2.unsqueeze(0), 
                                     dim=1)
        return cos_sim.item()
    
    def l2_distance(self, grad1, grad2):
        """Calculate L2 distance between gradients. Lower = more similar."""
        total_distance = 0
        for key in grad1.keys():
            if key in grad2:
                diff = grad1[key] - grad2[key]
                total_distance += torch.norm(diff).item()
        return total_distance
    
    def directional_alignment(self, grad1, grad2):
        """Measure how aligned the gradient directions are."""
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for key in grad1.keys():
            if key in grad2:
                flat1 = grad1[key].flatten()
                flat2 = grad2[key].flatten()
                dot_product += torch.dot(flat1, flat2).item()
                norm1 += torch.norm(flat1).item() ** 2
                norm2 += torch.norm(flat2).item() ** 2
        
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def layer_wise_similarity(self, grad1, grad2):
        """Calculate similarity for each layer separately."""
        layer_metrics = {}
        for key in grad1.keys():
            if key in grad2:
                # Cosine similarity for this layer
                flat1 = grad1[key].flatten()
                flat2 = grad2[key].flatten()
                cos_sim = F.cosine_similarity(
                    flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1
                ).item()
                
                # L2 distance for this layer
                l2_dist = torch.norm(grad1[key] - grad2[key]).item()
                
                layer_metrics[key] = {
                    'cosine_similarity': cos_sim,
                    'l2_distance': l2_dist,
                    'shape': tuple(grad1[key].shape)
                }
        
        return layer_metrics
    
    def gradient_norm_ratio(self, grad1, grad2):
        """Calculate the ratio of gradient norms."""
        norm1 = sum(torch.norm(g).item() for g in grad1.values())
        norm2 = sum(torch.norm(g).item() for g in grad2.values())
        
        if norm2 == 0:
            return float('inf')
        return norm1 / norm2
    
    def parameter_wise_correlation(self, grad1, grad2):
        """Calculate correlation between gradient parameters."""
        all_params1 = []
        all_params2 = []
        
        for key in grad1.keys():
            if key in grad2:
                all_params1.extend(grad1[key].flatten().tolist())
                all_params2.extend(grad2[key].flatten().tolist())
        
        if len(all_params1) < 2:
            return 0.0
            
        correlation = np.corrcoef(all_params1, all_params2)[0, 1]
        return 0.0 if np.isnan(correlation) else correlation
    
    def evaluate(self, original_grads, reconstructed_grads):
        """Comprehensive evaluation of gradient similarity."""
        return {
            'cosine_similarity': self.cosine_similarity(original_grads, reconstructed_grads),
            'l2_distance': self.l2_distance(original_grads, reconstructed_grads),
            'directional_alignment': self.directional_alignment(original_grads, reconstructed_grads),
            'gradient_norm_ratio': self.gradient_norm_ratio(original_grads, reconstructed_grads),
            'parameter_correlation': self.parameter_wise_correlation(original_grads, reconstructed_grads),
            'layer_wise_similarity': self.layer_wise_similarity(original_grads, reconstructed_grads)
        }


class AttackSuccessMetrics:
    """Metrics specific to attack success evaluation."""
    
    @staticmethod
    def privacy_leakage_score(original_grads, reconstructed_grads, threshold=0.7):
        """Calculate privacy leakage score based on gradient similarity.
        Higher score = more privacy leakage.
        """
        metrics = GradientQualityMetrics()
        cos_sim = metrics.cosine_similarity(original_grads, reconstructed_grads)
        
        # Higher cosine similarity = more privacy leakage
        # Scale from threshold to 1.0
        if cos_sim < threshold:
            leakage_score = 0.0
        else:
            leakage_score = (cos_sim - threshold) / (1.0 - threshold)
        
        return min(leakage_score, 1.0)
    
    @staticmethod
    def attack_success_rate(original_grads_list, reconstructed_grads_list, threshold=0.7):
        """Calculate attack success rate across multiple samples."""
        if not original_grads_list:
            return 0.0
            
        success_count = 0
        metrics = GradientQualityMetrics()
        
        for orig, recon in zip(original_grads_list, reconstructed_grads_list):
            cos_sim = metrics.cosine_similarity(orig, recon)
            if cos_sim >= threshold:
                success_count += 1
        
        return success_count / len(original_grads_list)
    
    @staticmethod
    def classify_privacy_risk(cosine_similarity):
        """Classify privacy risk based on cosine similarity."""
        if cosine_similarity >= 0.8:
            return "CRITICAL", "High privacy leakage - data reconstruction likely"
        elif cosine_similarity >= 0.6:
            return "HIGH", "Significant privacy leakage - features exposed"
        elif cosine_similarity >= 0.4:
            return "MEDIUM", "Moderate privacy leakage - some information exposed"
        elif cosine_similarity >= 0.2:
            return "LOW", "Minor privacy leakage - limited reconstruction"
        else:
            return "MINIMAL", "Good privacy preservation"


# !!!! HELPER FUNCTIONS FOR DATA FORMAT

def load_gradients_from_fl_output(filepath, client_id=0, step=0):
    """Load gradients from FL output format."""
    data = torch.load(filepath, map_location='cpu')
    
    if 'raw_gradients' not in data:
        raise ValueError("File doesn't contain raw_gradients")
    
    if client_id not in data['raw_gradients']:
        available = list(data['raw_gradients'].keys())
        raise ValueError(f"Client {client_id} not found. Available: {available}")
    
    client_data = data['raw_gradients'][client_id]
    
    # Return gradients from specific step
    if step >= len(client_data['grads_per_step_raw']):
        raise ValueError(f"Step {step} not available. Max step: {len(client_data['grads_per_step_raw'])-1}")
    
    return client_data['grads_per_step_raw'][step]

def compare_client_gradients(filepath, client1=0, client2=1, step=0):
    """Compare gradients between two clients."""
    grads1 = load_gradients_from_fl_output(filepath, client1, step)
    grads2 = load_gradients_from_fl_output(filepath, client2, step)
    
    metrics = GradientQualityMetrics()
    return metrics.evaluate(grads1, grads2)

def analyze_all_clients_in_round(filepath, step=0):
    """Analyze gradients for all clients in a round."""
    data = torch.load(filepath, map_location='cpu')
    results = {}
    metrics = GradientQualityMetrics()
    
    client_ids = list(data['raw_gradients'].keys())
    
    for i, client1 in enumerate(client_ids):
        for client2 in client_ids[i+1:]:
            grads1 = load_gradients_from_fl_output(filepath, client1, step)
            grads2 = load_gradients_from_fl_output(filepath, client2, step)
            
            key = f"client_{client1}_vs_client_{client2}"
            results[key] = metrics.evaluate(grads1, grads2)
    
    return results

def get_round_info(filepath):
    """Get basic information about a round file."""
    data = torch.load(filepath, map_location='cpu')
    info = {
        'file': str(filepath),
        'clients_available': list(data['raw_gradients'].keys()),
        'gradient_steps_per_client': {},
        'layer_names': []
    }
    
    if info['clients_available']:
        first_client = info['clients_available'][0]
        first_grads = data['raw_gradients'][first_client]['grads_per_step_raw']
        info['gradient_steps_per_client'][first_client] = len(first_grads)
        if first_grads:
            info['layer_names'] = list(first_grads[0].keys())
    
    return info

# !!! UTILITY FUNCTIONS FOR SAVING/LOADING GRADIENTS

def save_gradients(gradients, filepath):
    """Save gradients to .pt file."""
    # Ensure all tensors are on CPU before saving
    cpu_gradients = {k: v.cpu() for k, v in gradients.items()}
    torch.save(cpu_gradients, filepath)

def load_gradients(filepath, device='cpu'):
    """Load gradients from .pt file."""
    gradients = torch.load(filepath, map_location=device)
    return gradients

def gradients_to_device(gradients, device):
    """Move gradients to specified device."""
    return {k: v.to(device) for k, v in gradients.items()}

def save_gradient_metrics(metrics, filepath):
    """Save gradient metrics to JSON file."""
    # Convert any tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        return obj
    
    serializable_metrics = convert_tensors(metrics)
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)