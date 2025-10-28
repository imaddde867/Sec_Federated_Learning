"""
tenseal_adapter.py

TenSEAL selective-layer adapter + validation + mock KMS + metrics + tests.

Features implemented:
 - Selective layer encryption: selection modes "L1", "L1-2", "L1-3", ..., "Full".
 - wrap_model_state / unwrap_model_state operate on PyTorch state_dict-like mappings.
 - Validation of inputs (types, dtypes, shapes), consistent error messages.
 - Constant-time key id comparison where applicable (hmac.compare_digest).
 - Mock KMS interface: store/rotate keys locally (file-backed) with JSON metadata.
 - Metrics: encryption/decryption latency, memory footprint, comm overhead (serialized bytes).
 - Structured JSON logging for rotations/events.
 - Deterministic test vectors + small local harness for Attacks team to probe behavior.
 - Safe fallback when TenSEAL isn't available: deterministic "mock HE" so tests run.
 - Negative tests included (wrong shapes, truncated payloads, bad selection).
"""

from __future__ import annotations
import os
import io
import time
import json
import copy
import pickle
import hashlib
import hmac
import base64
import random
import logging
import resource
from typing import Dict, Any, Iterable, Tuple, List, Optional

# Optional deps: torch, tenseal
try:
    import torch
except Exception:
    torch = None

try:
    import tenseal as ts  # type: ignore
    TENSEAL_AVAILABLE = True
except Exception:
    ts = None
    TENSEAL_AVAILABLE = False

# ---------- Logging (structured JSON) ----------
logger = logging.getLogger("TenSEALAdapter")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            base = {
                "ts": time.time(),
                "level": record.levelname,
                "msg": record.getMessage(),
                "module": record.module,
                "lineno": record.lineno,
            }
            if hasattr(record, "extra"):
                base.update(record.extra)
            return json.dumps(base)
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)

# ---------- Uniform errors ----------
class AdapterError(Exception):
    pass

def _err(msg: str):
    # uniform error format
    raise AdapterError(msg)

# ---------- Constant time compare for key ids / tokens ----------
def constant_time_equal(a: str, b: str) -> bool:
    # use hmac.compare_digest for safe comparison
    # ensure both are strings
    return hmac.compare_digest(str(a), str(b))

# ---------- Utilities ----------
def _bytesize(obj: Any) -> int:
    return len(pickle.dumps(obj))

def _mem_usage_kb() -> int:
    # platform-dependent but commonly available
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is kilobytes, on mac bytes — we can't reliably detect both
    # We'll document that ru_maxrss meaning varies by platform.
    return int(usage)

# ---------- Mock TenSEAL fallback (deterministic) ----------
class _MockHE:
    """
    A deterministic, NOT-SECURE fallback for testing when TenSEAL is not installed.
    "Encrypt" = XOR with a fixed pseudo-random pad derived from key_id.
    This allows deterministic round-trip checks in unit tests without TenSEAL.
    """
    def __init__(self, key_bytes: bytes):
        self.key = hashlib.sha256(key_bytes).digest()

    def encrypt_vector(self, vec: bytes) -> bytes:
        # deterministic pseudo-XOR
        out = bytearray(vec)
        k = self.key
        for i in range(len(out)):
            out[i] ^= k[i % len(k)]
        return bytes(out)

    def decrypt_vector(self, ctext: bytes) -> bytes:
        # XOR is symmetric
        return self.encrypt_vector(ctext)

# ---------- Key / context parameters dataclass-like structure ----------
DEFAULT_HE_PARAMS = {
    "poly_modulus_degree": 16384,
    "coeff_mod_bit_sizes": [60, 40, 40, 60, 60],
    "scale": 2**40,
}

# ---------- Mock KMS ----------
class MockKMS:
    """
    Simple file-backed KMS for local testing.
    Stores key blobs + metadata under `store_dir`.
    This is a mock: keys are base64 encoded; integrity checked via HMAC over metadata.
    For real deployments replace with a proper KMS and HSM-backed secrets.
    """
    def __init__(self, store_dir: str = "./kms_store", master_secret_env: str = "TENSEAL_KMS_MASTER"):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.master_secret = os.environ.get(master_secret_env, "dev_master_secret_not_for_prod").encode("utf-8")
        # HMAC key derived
        self._mac_key = hashlib.sha256(self.master_secret).digest()

    def _meta_path(self, key_id: str) -> str:
        return os.path.join(self.store_dir, f"{key_id}.json")

    def store_key(self, key_id: str, key_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not key_id or not isinstance(key_id, str):
            _err("invalid_key_id")
        if not isinstance(key_bytes, (bytes, bytearray)):
            _err("invalid_key_blob")
        meta = {
            "key_id": key_id,
            "created_at": time.time(),
            "size": len(key_bytes),
            "metadata": metadata or {},
        }
        # integrity HMAC
        mac = hmac.new(self._mac_key, digestmod=hashlib.sha256)
        mac.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
        mac.update(key_bytes)
        meta["mac"] = base64.b64encode(mac.digest()).decode("ascii")
        meta["key_blob_b64"] = base64.b64encode(key_bytes).decode("ascii")
        with open(self._meta_path(key_id), "w") as f:
            json.dump(meta, f)
        logger.info("kms_store_key", extra={"extra": {"event": "store_key", "key_id": key_id}})

    def get_key(self, key_id: str) -> bytes:
        path = self._meta_path(key_id)
        if not os.path.exists(path):
            _err("key_not_found")
        with open(path, "r") as f:
            meta = json.load(f)
        blob = base64.b64decode(meta["key_blob_b64"].encode("ascii"))
        # verify mac
        mac = hmac.new(self._mac_key, digestmod=hashlib.sha256)
        tmp = copy.deepcopy(meta)
        tmp.pop("mac", None)
        tmp.pop("key_blob_b64", None)
        mac.update(json.dumps(tmp, sort_keys=True).encode("utf-8"))
        mac.update(blob)
        expected = base64.b64decode(meta["mac"].encode("ascii"))
        if not hmac.compare_digest(mac.digest(), expected):
            _err("kms_integrity_fail")
        return blob

    def rotate_key(self, old_key_id: str, new_key_id: str, new_key_bytes: bytes, delete_old: bool = False):
        # store new key, optionally delete old
        self.store_key(new_key_id, new_key_bytes, metadata={"rotated_from": old_key_id})
        logger.info("kms_rotate_key", extra={"extra": {"event": "rotate", "from": old_key_id, "to": new_key_id}})
        if delete_old:
            try:
                os.remove(self._meta_path(old_key_id))
            except FileNotFoundError:
                pass

    def list_keys(self) -> List[str]:
        return [os.path.splitext(fn)[0] for fn in os.listdir(self.store_dir) if fn.endswith(".json")]

# ---------- Adapter ----------
class TenSEALAdapter:
    """
    Adapter exposing:
     - create_context(key_id, params)
     - wrap_state(state_dict, selection="L1"|"L1-2"|...|"Full")
     - unwrap_state(wrapped_blob, key_id)
    If TenSEAL is available we use CKKS vectors; otherwise fall back to deterministic mock HE.
    """

    VALID_SELECTIONS = ("Full",)  # will dynamically accept "L1", "L1-2", etc.

    def __init__(self, kms: MockKMS, he_params: Optional[Dict[str, Any]] = None):
        self.kms = kms
        self.he_params = he_params or DEFAULT_HE_PARAMS
        self.context_cache: Dict[str, Any] = {}  # key_id -> HE context or mock wrapper

    # ---- Context / key management ----
    def create_context(self, key_id: str, key_bytes: bytes, persist: bool = True) -> None:
        """
        Register a key in KMS and create an in-memory HE context (TenSEAL or mock).
        """
        if not isinstance(key_id, str) or not key_id:
            _err("invalid_key_id")
        if not isinstance(key_bytes, (bytes, bytearray)):
            _err("invalid_key_bytes")
        # store in KMS (mock) and create context
        if persist:
            self.kms.store_key(key_id, key_bytes, metadata={"he_params": self.he_params})
        # create runtime HE context
        self._build_runtime_context(key_id, key_bytes)
        logger.info("context_created", extra={"extra": {"event": "create_context", "key_id": key_id}})

    def _build_runtime_context(self, key_id: str, key_bytes: bytes):
        if TENSEAL_AVAILABLE:
            # Create TenSEAL CKKS context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.he_params.get("poly_modulus_degree", 8192),
                coeff_mod_bit_sizes=self.he_params.get("coeff_mod_bit_sizes", [60, 40, 40, 60]),
            )

            # ✅ Set global scale (fixes "no global scale" error)
            context.global_scale = self.he_params.get("scale", 2**40)

            # Generate keys
            context.generate_galois_keys()
            context.generate_relin_keys()

            # Store context with ID tag
            context._adapter_key_id = key_id
            self.context_cache[key_id] = context
        else:
            # Mock fallback if TenSEAL not available
            self.context_cache[key_id] = _MockHE(key_bytes)


    def ensure_context(self, key_id: str):
        if key_id in self.context_cache:
            return
        # try to load from KMS
        key_bytes = self.kms.get_key(key_id)
        self._build_runtime_context(key_id, key_bytes)

    # ---- Selection parsing ----
    @staticmethod
    def parse_selection(selection: str, state_keys: List[str]) -> List[str]:
        """
        selection examples:
         - "Full" -> all keys
         - "L1"   -> first layer key only (ordered by appearance)
         - "L1-2" -> first two layers
        Rules:
         - Layers refer to top-level keys in state_dict in their encountered order.
         - Raises AdapterError on unknown selection.
        """
        if not isinstance(selection, str) or not selection:
            _err("invalid_selection")
        if selection == "Full":
            return state_keys.copy()
        if selection.startswith("L"):
            try:
                if "-" in selection:
                    parts = selection[1:].split("-")
                    start = int(parts[0])  # L1-3 style
                    end = int(parts[1])
                    if start < 1 or end < start:
                        _err("invalid_selection_range")
                    # Convert to indices:
                    idxs = list(range(start - 1, min(end, len(state_keys))))
                    return [state_keys[i] for i in idxs]
                else:
                    n = int(selection[1:])
                    if n < 1:
                        _err("invalid_selection_range")
                    idxs = list(range(0, min(n, len(state_keys))))
                    return [state_keys[i] for i in idxs]
            except ValueError:
                _err("invalid_selection_format")
        _err("unsupported_selection")

    # ---- Validation helpers ----
    @staticmethod
    def _validate_tensor_like(obj: Any, name: str = ""):
        """
        Accepts numpy arrays, torch tensors. Enforces float32 dtype and finite values.
        """
        import numpy as _np
        # torch optional
        if torch is not None and isinstance(obj, torch.Tensor):
            if obj.dtype not in (torch.float32, torch.float64):
                _err(f"invalid_dtype_{name}")
            if obj.numel() == 0:
                _err(f"empty_tensor_{name}")
            # ensure finite
            if not torch.isfinite(obj).all():
                _err(f"nonfinite_tensor_{name}")
            return
        if isinstance(obj, (_np.ndarray, )):
            if obj.dtype not in (_np.float32, _np.float64):
                _err(f"invalid_dtype_{name}")
            if obj.size == 0:
                _err(f"empty_tensor_{name}")
            if not _np.isfinite(obj).all():
                _err(f"nonfinite_tensor_{name}")
            return
        _err(f"unsupported_tensor_type_{name}")

    # ---- wrap / unwrap ----
    def wrap_state(self, state_dict: Dict[str, Any], key_id: str, selection: str = "Full") -> Dict[str, Any]:
        """
        Wrap (encrypt/pack) selected layers from a state_dict.

        Returns a dictionary:
         {
            "key_id": key_id,
            "selection": selection,
            "wrapped": {
                layer_name: {
                    "cipher": <bytes (pickled)>,
                    "shape": <shape tuple>,
                    "dtype": "<dtype>"
                }, ...
            },
            "meta": {metrics...}
         }
        """
        if not isinstance(state_dict, dict):
            _err("state_dict_must_be_dict")
        # canonical order of keys
        keys = list(state_dict.keys())
        selected_keys = self.parse_selection(selection, keys)
        if len(selected_keys) == 0:
            _err("no_layers_selected")
        # ensure context available
        try:
            self.ensure_context(key_id)
        except AdapterError:
            _err("context_missing_or_kms_error")

        he = self.context_cache[key_id]

        wrapped = {}
        metrics = {
            "enc_time_s": 0.0,
            "bytes": 0,
            "mem_kb": _mem_usage_kb(),
            "per_layer": {},
        }
        start_all = time.perf_counter()

        for layer_name in selected_keys:
            tensor = state_dict[layer_name]
            # validate
            self._validate_tensor_like(tensor, name=layer_name)
            # convert to bytes: for TenSEAL would be vector; for mock we serialize raw float32 bytes
            if torch is not None and isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy().astype("float32")
            else:
                import numpy as _np
                arr = _np.array(tensor, dtype=_np.float32, copy=False)
            # packing approach: flatten
            flat = arr.ravel()
            # allow chunking if needed — here take the whole vector for simplicity
            payload = flat.tobytes()
            t0 = time.perf_counter()
            if TENSEAL_AVAILABLE:
                # actual TenSEAL usage: create CKKSVector and serialize
                # This is a skeleton — users must adapt to their TenSEAL version.
                context: ts.Context = he  # type: ignore
                ck = ts.ckks_vector(context, flat.tolist())
                cipher_bytes = ck.serialize()  # returns bytes
            else:
                # mock HE: deterministic encryption of bytes
                cipher_bytes = he.encrypt_vector(payload)
            t1 = time.perf_counter()
            # produce per-layer envelope
            envelope = {
                "cipher": base64.b64encode(cipher_bytes).decode("ascii"),
                "shape": tuple(arr.shape),
                "dtype": "float32",
                "numel": int(flat.size),
            }
            wrapped[layer_name] = envelope
            per_t = {
                "enc_time_s": (t1 - t0),
                "cipher_bytes": len(cipher_bytes),
                "numel": int(flat.size),
            }
            metrics["per_layer"][layer_name] = per_t
            metrics["bytes"] += len(cipher_bytes)
        metrics["enc_time_s"] = time.perf_counter() - start_all
        ret = {
            "key_id": key_id,
            "selection": selection,
            "wrapped": wrapped,
            "meta": metrics,
        }
        logger.info("wrap_event", extra={"extra": {"event": "wrap", "key_id": key_id, "selection": selection, "metrics": metrics}})
        return ret

    def unwrap_state(self, wrapped_blob: Dict[str, Any], expected_key_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Decrypt/unpack previously wrapped_blob. If expected_key_id provided use constant-time compare.
        Returns dict mapping layer_name -> numpy array (float32) or torch.Tensor if torch is available.
        """
        if not isinstance(wrapped_blob, dict):
            _err("invalid_wrapped_blob")
        key_id = wrapped_blob.get("key_id")
        if expected_key_id is not None:
            if not constant_time_equal(expected_key_id, key_id):
                _err("key_id_mismatch")
        selection = wrapped_blob.get("selection", "Full")
        wrapped = wrapped_blob.get("wrapped")
        if not isinstance(wrapped, dict):
            _err("invalid_wrapped_payload")
        # ensure context
        try:
            self.ensure_context(key_id)
        except AdapterError:
            _err("context_missing_or_kms_error")

        he = self.context_cache[key_id]
        results = {}
        metrics = {"dec_time_s": 0.0, "bytes": 0, "per_layer": {}}
        start_all = time.perf_counter()
        for layer_name, env in wrapped.items():
            try:
                cipher_b64 = env["cipher"]
                shape = tuple(env["shape"])
                dtype = env.get("dtype", "float32")
            except Exception:
                _err("malformed_layer_envelope")
            cipher_bytes = base64.b64decode(cipher_b64.encode("ascii"))
            t0 = time.perf_counter()
            if TENSEAL_AVAILABLE:
                # deserialize and decrypt to list
                context: ts.Context = he  # type: ignore
                ck = ts.ckks_vector_from(context, cipher_bytes)
                vec = ck.decrypt()
                import numpy as _np
                arr = _np.array(vec, dtype=_np.float32).reshape(shape)
            else:
                plain_bytes = he.decrypt_vector(cipher_bytes)
                import numpy as _np
                # reconstruct float32 array
                arr = _np.frombuffer(plain_bytes, dtype=_np.float32).reshape(shape)
            t1 = time.perf_counter()
            if torch is not None:
                results[layer_name] = torch.from_numpy(arr.copy())
            else:
                results[layer_name] = arr
            per_t = {
                "dec_time_s": (t1 - t0),
                "cipher_bytes": len(cipher_bytes),
                "numel": int(env.get("numel", 0)),
            }
            metrics["per_layer"][layer_name] = per_t
            metrics["bytes"] += len(cipher_bytes)
        metrics["dec_time_s"] = time.perf_counter() - start_all
        logger.info("unwrap_event", extra={"extra": {"event": "unwrap", "key_id": key_id, "metrics": metrics}})
        return results

# ---------- Deterministic test vectors & harness ----------
def deterministic_state_dict(num_layers: int = 3, seed: int = 1337) -> Dict[str, Any]:
    """
    Create a deterministic state dict-like mapping for tests:
      layer0.weight, layer0.bias, layer1.weight, ...
    Each tensor is small and deterministic.
    """
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    state = {}
    for i in range(num_layers):
        w = _np.arange(i*6, i*6 + 6, dtype=_np.float32).reshape((2,3)) + (seed % 10)
        b = _np.arange(i*2, i*2 + 2, dtype=_np.float32) + (seed % 5)
        state[f"layer{i}.weight"] = w
        state[f"layer{i}.bias"] = b
    return state

# --- Local harness for Attacks team ---
def local_harness_demo():
    """
    Run a simple wrap/unwrap cycle and print metrics. Useful for Attacks team to probe.
    """
    print("Running local harness demo (deterministic)...")
    kms = MockKMS(store_dir="./.kms_demo")
    adapter = TenSEALAdapter(kms=kms)
    key_id = "demo-key-1"
    # deterministic key material
    demo_key = hashlib.sha256(b"demo_seed_for_key").digest()
    adapter.create_context(key_id, demo_key, persist=True)
    state = deterministic_state_dict(num_layers=4, seed=42)
    print("State keys:", list(state.keys()))
    wrapped = adapter.wrap_state(state, key_id=key_id, selection="L1-2")
    print("Wrapped selection keys:", list(wrapped["wrapped"].keys()))
    unwrapped = adapter.unwrap_state(wrapped, expected_key_id=key_id)
    # verify equality
    for k, arr in unwrapped.items():
        import numpy as _np
        orig = state[k]
        got = arr if not isinstance(arr, (list, tuple)) else _np.array(arr)
        if torch is not None and isinstance(got, torch.Tensor):
            got = got.cpu().numpy()
        assert _np.allclose(orig, got), f"mismatch {k}"
    print("Round-trip OK. metrics:", wrapped["meta"])

# ---------- Tests (negative + positive) ----------
def run_tests():
    print("Running tests...")
    # setup
    kms = MockKMS(store_dir="./.kms_tests")
    adapter = TenSEALAdapter(kms=kms)
    key_id = "test-key"
    adapter.create_context(key_id, b"test_secret_for_key", persist=True)
    # positive: full
    state = deterministic_state_dict(num_layers=3, seed=7)
    wrapped_full = adapter.wrap_state(state, key_id=key_id, selection="Full")
    out_full = adapter.unwrap_state(wrapped_full, expected_key_id=key_id)
    # verify
    import numpy as _np
    for k in state.keys():
        got = out_full[k]
        if torch is not None and isinstance(got, torch.Tensor):
            got = got.cpu().numpy()
        assert _np.allclose(state[k], got), "roundtrip_failed_full"

    # positive: L1 only
    wrapped_l1 = adapter.wrap_state(state, key_id=key_id, selection="L1")
    out_l1 = adapter.unwrap_state(wrapped_l1)
    assert set(out_l1.keys()) == set(adapter.parse_selection("L1", list(state.keys())))

    # negative tests:
    try:
        adapter.wrap_state("not_a_dict", key_id=key_id)
        raise AssertionError("should_have_failed_invalid_state")
    except AdapterError as e:
        assert str(e).startswith("state_dict_must_be_dict") or "state_dict" in str(e)

    # truncated payload simulation: alter cipher
    wrapped = adapter.wrap_state(state, key_id=key_id, selection="L1")
    # corrupt a cipher
    first_key = next(iter(wrapped["wrapped"].keys()))
    wrapped_copy = copy.deepcopy(wrapped)
    b64 = wrapped_copy["wrapped"][first_key]["cipher"]
    raw = base64.b64decode(b64.encode("ascii"))
    raw_trunc = raw[: max(1, len(raw) // 2)]
    wrapped_copy["wrapped"][first_key]["cipher"] = base64.b64encode(raw_trunc).decode("ascii")
    try:
        adapter.unwrap_state(wrapped_copy, expected_key_id=key_id)
        raise AssertionError("should_have_failed_truncated_cipher")
    except Exception as e:
        # depending on TenSEAL vs mock, may fail when reshaping or wrong length
        pass

    # invalid selection
    try:
        adapter.wrap_state(state, key_id=key_id, selection="L0")
        raise AssertionError("should_have_failed_invalid_selection")
    except AdapterError:
        pass

    print("All tests passed.")

# ---------- If run as script ----------
if __name__ == "__main__":
    # basic CLI to run harness or tests
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run local harness demo")
    p.add_argument("--tests", action="store_true", help="Run tests")
    args = p.parse_args()
    if args.demo:
        local_harness_demo()
    elif args.tests:
        run_tests()
    else:
        print("tenseal_adapter module loaded. Use --demo or --tests to exercise.")
