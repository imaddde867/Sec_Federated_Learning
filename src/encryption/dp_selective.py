from typing import Dict, Any, List
import torch

class DPSelectiveEncryptor:
    """Clip and add Gaussian noise to selected parameter deltas.
    layers: 'None','L1','L1-2','L1-3','Full'
    """
    def __init__(self, layers: str="L1", sigma: float=1.0, clip: float=1.0):
        self.layers = layers
        self.sigma = float(sigma)
        self.clip = float(clip)

    def parameters_to_encrypt(self, named_tensors: Dict[str, torch.Tensor]) -> List[str]:
        if self.layers == "None": return []
        if self.layers == "Full": return list(named_tensors.keys())
        max_layer = {"L1":1, "L1-2":2, "L1-3":3}.get(self.layers, 1)
        protect = []
        for n in named_tensors:
            for L in range(1, max_layer+1):
                if n.startswith(f"layer{L}") or f"features.{L-1}" in n:  # cover VGG-style too
                    protect.append(n); break
        return protect

    def encrypt_state_dict(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        protected = self.parameters_to_encrypt(delta)
        out = {}
        for k, v in delta.items():
            if k in protected and v is not None and torch.is_tensor(v):
                # per-tensor clipping (L2) and Gaussian noise
                flat = v.view(-1)
                norm = torch.norm(flat, p=2) + 1e-12
                scale = min(1.0, self.clip / norm.item())
                v_clip = v * scale
                noise = torch.randn_like(v) * self.sigma
                out[k] = v_clip + noise
            else:
                out[k] = v
        return out
