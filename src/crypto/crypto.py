from __future__ import annotations
from typing import Literal, Tuple
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
import torch
import numpy as np

Mode = Literal["aes-gcm-128", "aes-gcm-256", "chacha20-poly1305"]

@dataclass
class Encryptor:
    key: bytes
    mode: Mode = "aes-gcm-256"

    @staticmethod
    def generate(mode: Mode = "aes-gcm-256") -> "Encryptor":
        if mode == "aes-gcm-128":
            key = AESGCM.generate_key(bit_length=128)
        elif mode == "aes-gcm-256":
            key = AESGCM.generate_key(bit_length=256)
        elif mode == "chacha20-poly1305":
            key = ChaCha20Poly1305.generate_key()
        else:
            raise ValueError(f"Unsupported mode {mode}")
        return Encryptor(key=key, mode=mode)

    def _aead(self):
        if self.mode.startswith("aes-gcm"):
            return AESGCM(self.key)
        if self.mode == "chacha20-poly1305":
            return ChaCha20Poly1305(self.key)
        raise ValueError(self.mode)

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext and return (nonce, ciphertext). AAD is optional.
        """
        aead = self._aead()
        # 96-bit nonce for AES-GCM and ChaCha20-Poly1305
        nonce = os.urandom(12)
        ct = aead.encrypt(nonce, plaintext, aad)
        return nonce, ct

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        aead = self._aead()
        return aead.decrypt(nonce, ciphertext, aad)

def tensor_serialize(t: torch.Tensor) -> bytes:
    """
    Serialize a tensor to bytes (row-major). Shape/dtype must be carried out-of-band.
    """
    return t.detach().cpu().numpy().tobytes()

def tensor_deserialize(buf: bytes, like: torch.Tensor) -> torch.Tensor:
    """
    Deserialize bytes into a tensor shaped/dtyped like `like`.
    """
    arr = np.frombuffer(buf, dtype=like.detach().cpu().numpy().dtype)
    return torch.from_numpy(arr.reshape(like.shape))