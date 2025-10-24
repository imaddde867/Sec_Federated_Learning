from typing import Dict, Any, List

class DPSelectiveEncryptor:
    """
    Per-layer gradient clipping + Gaussian noise (placeholder; integrate with tensor ops).
    layers: 'None','L1','L1-2','L1-3','Full'
    """
    def __init__(self, layers: str="L1", sigma: float=1.0, clip: float=1.0):
        self.layers = layers
        self.sigma = sigma
        self.clip = clip

    def parameters_to_encrypt(self, named_grads: Dict[str, Any]) -> List[str]:
        if self.layers == "None": return []
        if self.layers == "Full": return list(named_grads.keys())
        max_layer = {"L1":1, "L1-2":2, "L1-3":3}.get(self.layers, 1)
        protect = []
        for n in named_grads:
            for L in range(1, max_layer+1):
                if n.startswith(f"layer{L}"):
                    protect.append(n); break
        return protect

    def encrypt(self, named_grads: Dict[str, Any]):
        # TODO: implement clipping and add Gaussian noise
        return self.parameters_to_encrypt(named_grads)
