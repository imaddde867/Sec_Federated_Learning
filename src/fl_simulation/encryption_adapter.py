import torch
from typing import Dict, List, Optional, Callable, Any, Union

class SelectiveEncryptionAdapter:

    def __init__(
        self,
        layers_to_encrypt: Optional[List[str]] = None,
        mode: str = "noop",
        name_filter: Optional[Callable[[str], bool]] = None,
        drop_plaintext_when_encrypted: bool = False,
        he_ctx: Any = None,  # placeholder for TenSEAL context/keys
    ):
        self.layers_to_encrypt = layers_to_encrypt or []
        self.enabled = len(self.layers_to_encrypt) > 0
        self.mode = (mode or "noop").lower()
        self.name_filter = name_filter  # extra filter on param names
        self.drop_plaintext_when_encrypted = bool(drop_plaintext_when_encrypted)
        self.he_ctx = he_ctx  # future: TenSEAL context/keys

        if self.mode not in ("noop", "he", "mask"):
            raise ValueError(f"Unsupported mode: {self.mode}")

    # selection

    def should_encrypt(self, param_name: str) -> bool:
        if not self.enabled:
            return False
        layer_hit = any(layer in param_name for layer in self.layers_to_encrypt)
        if self.name_filter is not None:
            layer_hit = layer_hit and bool(self.name_filter(param_name))
        return layer_hit

    # wrappers / serialization

    def _wrap_entry(
        self,
        data: Union[torch.Tensor, Any],
        encrypted: bool,
        scheme: str = "none",
        meta: Optional[dict] = None,
    ) -> dict:
        meta = meta or {}
        t = data.detach().clone() if isinstance(data, torch.Tensor) else data
        entry = {
            "encrypted": encrypted,
            "scheme": scheme,          # 'none' | 'ckks' | 'mask' ...
            "data": t,
            "original_shape": tuple(t.shape) if isinstance(t, torch.Tensor) else None,
            "dtype": str(t.dtype) if isinstance(t, torch.Tensor) else None,
            "device": str(t.device) if isinstance(t, torch.Tensor) else None,
        }
        if meta:
            entry["meta"] = dict(meta)
        return entry

    def serialize(self, entry: dict) -> dict:
        """Placeholder for wire/storage format; keep as is for now."""
        return entry

    def deserialize(self, entry: dict) -> dict:
        """Placeholder; if HE objects need special hooks, add here."""
        return entry

    # tensor-level ops (placeholders for HE/masking)

    def _encrypt_tensor(self, name: str, t: torch.Tensor) -> dict:
        if self.mode == "noop":
            return self._wrap_entry(t, encrypted=False, scheme="none")
        if self.mode == "he":
            # TODO: Replace with real HE encrypt (e.g., TenSEAL CKKS)
            # Example (pseudo):
            #   vec = t.detach().cpu().flatten().numpy().astype(np.float64)
            #   ct = ts.ckks_vector(self.he_ctx, vec)
            #   return self._wrap_entry(ct, encrypted=True, scheme="ckks")
            return self._wrap_entry(t, encrypted=True, scheme="ckks")  # placeholder
        if self.mode == "mask":
            # TODO: client-side generate random mask r, send (t+r), server sums, then
            # masks cancel in aggregation (requires protocol among clients).
            return self._wrap_entry(t, encrypted=True, scheme="mask")
        raise RuntimeError("Unknown mode")

    def _aggregate_entries(self, entries: List[dict]) -> dict:
        """Aggregate encrypted entries; fallback to mean for noop."""
        if not entries:
            return {}
        scheme = entries[0].get("scheme", "none")
        if self.mode == "noop" or scheme == "none":
            # mean of plaintext tensors
            acc = None
            for e in entries:
                x = e["data"]
                acc = x.clone() if acc is None else acc + x
            acc = acc / float(len(entries))
            return self._wrap_entry(acc, encrypted=False, scheme="none")
        if self.mode == "he":
            # TODO: do HE addition on ciphertexts, then client-side/server-side decryption
            # For now: pass-through first one (placeholder)
            return entries[0]
        if self.mode == "mask":
            # TODO: sum (masked) then demask (requires protocol and mask bookkeeping)
            return entries[0]
        raise RuntimeError("Unknown mode in aggregation")

    # public APIs used by your trainer

    def encrypt_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for name, grad in gradients.items():
            if self.should_encrypt(name):
                entry = self._encrypt_tensor(name, grad)
                out[name] = self.serialize(entry)
                if entry["encrypted"]:
                    print(f"Encrypted grad: {name} [{entry['scheme']}]")
            else:
                entry = self._wrap_entry(grad, encrypted=False, scheme="none")
                out[name] = self.serialize(entry)
        return out

    def encrypt_update(self, delta: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for name, tensor in delta.items():
            if self.should_encrypt(name):
                entry = self._encrypt_tensor(name, tensor)
                out[name] = self.serialize(entry)
                if entry["encrypted"]:
                    print(f"Encrypted update: {name} [{entry['scheme']}]")
            else:
                entry = self._wrap_entry(tensor, encrypted=False, scheme="none")
                out[name] = self.serialize(entry)
        return out

    def decrypt_and_aggregate(self, encrypted_grads_list: List[Dict[str, dict]]) -> Dict[str, torch.Tensor]:
        """Return plaintext mean if mode='noop'; otherwise placeholder behavior."""
        if not encrypted_grads_list:
            return {}
        # group by parameter name
        grouped: Dict[str, List[dict]] = {}
        for enc in encrypted_grads_list:
            for name, entry in enc.items():
                grouped.setdefault(name, []).append(self.deserialize(entry))
        # aggregate per-parameter
        result: Dict[str, torch.Tensor] = {}
        for name, entries in grouped.items():
            agg_entry = self._aggregate_entries(entries)
            # If ciphertext: you would insert decrypt here.
            if agg_entry["encrypted"]:
                # TODO: decrypt; currently return placeholder tensor for compatibility
                # For now, keep the data field (assumed tensor in noop).
                pass
            result[name] = agg_entry["data"] if isinstance(agg_entry["data"], torch.Tensor) \
                           else torch.tensor(0.0)  # safe fallback
        return result

