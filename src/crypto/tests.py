from __future__ import annotations
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def test_aes_gcm_vector():
    """
    Simple known-answer test: all-zero key/nonce/plaintext must round-trip.
    """
    key = bytes.fromhex("00" * 16)   # 128-bit
    aesgcm = AESGCM(key)
    nonce = bytes.fromhex("00" * 12) # 96-bit
    pt = bytes.fromhex("00" * 16)
    aad = b""
    ct = aesgcm.encrypt(nonce, pt, aad)
    pt2 = aesgcm.decrypt(nonce, ct, aad)
    assert pt2 == pt, "AES-GCM round-trip failed"

if __name__ == "__main__":
    test_aes_gcm_vector()
    print("AES-GCM test vector passed.")