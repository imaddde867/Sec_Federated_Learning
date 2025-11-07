"""
Comprehensive Homomorphic Encryption Adapter for Federated Learning

Supports multiple HE schemes:
- TenSEAL CKKS (approximate arithmetic, good for ML)
- TenSEAL BFV (exact arithmetic)
- Mock HE (for testing without HE libraries)
- Simple additive HE (Paillier-like for aggregation)

Features:
- Selective layer encryption
- Client-server communication encryption
- Configurable encryption methods
- Performance metrics
"""

from __future__ import annotations
import os
import time
import json
import copy
import hashlib
import base64
import logging
import gc
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False

# Logging setup
logger = logging.getLogger("EncryptionAdapter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class EncryptionError(Exception):
    """Custom exception for encryption operations"""
    pass


class BaseHEAdapter:
    """Base class for homomorphic encryption adapters"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {
            "encryption_time": 0.0,
            "decryption_time": 0.0,
            "total_encrypted_bytes": 0,
            "num_encryptions": 0,
            "num_decryptions": 0,
        }
    
    def encrypt_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Any:
        """Encrypt a single tensor"""
        raise NotImplementedError
    
    def decrypt_tensor(self, encrypted: Any, shape: Tuple, dtype: str = "float32") -> Union[torch.Tensor, np.ndarray]:
        """Decrypt a single tensor"""
        raise NotImplementedError
    
    def encrypt_dict(self, tensor_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt a dictionary of tensors"""
        raise NotImplementedError
    
    def decrypt_dict(self, encrypted_dict: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Decrypt a dictionary of encrypted tensors"""
        raise NotImplementedError
    
    def aggregate_encrypted(self, encrypted_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate multiple encrypted updates (homomorphic addition)"""
        raise NotImplementedError
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "encryption_time": 0.0,
            "decryption_time": 0.0,
            "total_encrypted_bytes": 0,
            "num_encryptions": 0,
            "num_decryptions": 0,
        }


class MockHEAdapter(BaseHEAdapter):
    """
    Mock HE adapter for testing without actual HE libraries.
    Uses XOR with deterministic key for "encryption".
    WARNING: NOT SECURE - for testing only!
    """
    
    def __init__(self, enabled: bool = True, key: Optional[bytes] = None):
        super().__init__(enabled)
        if key is None:
            key = b"mock_he_key_for_testing_only"
        self.key = hashlib.sha256(key).digest()
        self.key_stream = self._generate_key_stream()
    
    def _generate_key_stream(self):
        """Generate deterministic key stream"""
        rng = np.random.RandomState(seed=42)
        return rng
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert to numpy array"""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().astype(np.float32)
        return np.array(tensor, dtype=np.float32, copy=False)
    
    def _to_torch(self, arr: np.ndarray) -> torch.Tensor:
        """Convert to torch tensor"""
        if TORCH_AVAILABLE:
            return torch.from_numpy(arr).float()
        return arr
    
    def encrypt_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Mock encryption: XOR with deterministic noise"""
        if not self.enabled:
            return {"data": tensor, "encrypted": False}
        
        arr = self._to_numpy(tensor)
        flat = arr.ravel()
        
        # Generate deterministic "noise" based on data hash
        data_hash = hashlib.sha256(flat.tobytes()).digest()[:16]
        seed = int.from_bytes(data_hash, 'big') % (2**31)
        rng = np.random.RandomState(seed=seed)
        noise = rng.randn(*flat.shape).astype(np.float32) * 0.01
        
        # "Encrypt" by adding noise (simulating HE)
        encrypted_flat = flat + noise
        
        start_time = time.perf_counter()
        self.metrics["encryption_time"] += time.perf_counter() - start_time
        self.metrics["num_encryptions"] += 1
        self.metrics["total_encrypted_bytes"] += encrypted_flat.nbytes
        
        return {
            "encrypted": True,
            "data": base64.b64encode(encrypted_flat.tobytes()).decode('ascii'),
            "shape": list(arr.shape),
            "dtype": "float32",
            "noise_seed": seed,
        }
    
    def decrypt_tensor(self, encrypted: Dict[str, Any], shape: Optional[Tuple] = None, dtype: str = "float32") -> Union[torch.Tensor, np.ndarray]:
        """Mock decryption"""
        if not encrypted.get("encrypted", False):
            return encrypted.get("data")
        
        data_bytes = base64.b64decode(encrypted["data"])
        arr = np.frombuffer(data_bytes, dtype=np.float32)
        shape = tuple(encrypted.get("shape", shape or (len(arr),)))
        arr = arr.reshape(shape)
        
        start_time = time.perf_counter()
        self.metrics["decryption_time"] += time.perf_counter() - start_time
        self.metrics["num_decryptions"] += 1
        
        return self._to_torch(arr)
    
    def encrypt_dict(self, tensor_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt dictionary of tensors"""
        if not self.enabled:
            return tensor_dict
        
        encrypted = {}
        for key, tensor in tensor_dict.items():
            encrypted[key] = self.encrypt_tensor(tensor)
        return encrypted
    
    def decrypt_dict(self, encrypted_dict: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Decrypt dictionary of encrypted tensors"""
        if not self.enabled:
            return encrypted_dict
        
        decrypted = {}
        for key, enc_data in encrypted_dict.items():
            if isinstance(enc_data, dict) and enc_data.get("encrypted"):
                decrypted[key] = self.decrypt_tensor(enc_data)
            else:
                decrypted[key] = enc_data
        return decrypted
    
    def aggregate_encrypted(self, encrypted_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate encrypted updates (homomorphic addition)"""
        if not encrypted_list:
            return {}
        
        if weights is None:
            weights = [1.0 / len(encrypted_list)] * len(encrypted_list)
        
        # Get all keys
        all_keys = set()
        for enc_dict in encrypted_list:
            all_keys.update(enc_dict.keys())
        
        aggregated = {}
        for key in all_keys:
            # Decrypt, aggregate, re-encrypt (mock HE - in real HE this would be homomorphic)
            decrypted_values = []
            for enc_dict, weight in zip(encrypted_list, weights):
                if key in enc_dict:
                    val = self.decrypt_tensor(enc_dict[key])
                    decrypted_values.append((val, weight))
            
            if decrypted_values:
                # Weighted sum
                result = None
                for val, weight in decrypted_values:
                    arr = self._to_numpy(val)
                    weighted = arr * weight
                    if result is None:
                        result = weighted
                    else:
                        result += weighted
                
                # Re-encrypt aggregated result
                aggregated[key] = self.encrypt_tensor(result)
        
        return aggregated


class TenSEALCKKSAdapter(BaseHEAdapter):
    """
    TenSEAL CKKS adapter for approximate homomorphic encryption.
    Good for machine learning workloads with floating point operations.
    """
    
    def __init__(self, enabled: bool = True, poly_modulus_degree: int = 8192, 
                 coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60], 
                 global_scale: float = 2**40, key: Optional[bytes] = None):
        super().__init__(enabled)
        
        if not TENSEAL_AVAILABLE:
            logger.warning("TenSEAL not available, falling back to MockHEAdapter")
            self._fallback = MockHEAdapter(enabled, key)
            self.tenseal_available = False
            return
        
        self.tenseal_available = True
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        
        # Generate or use provided key
        if key is None:
            key = os.urandom(32)
        self.key = key
        
        # Create context
        self.context = self._create_context()
        self.context.make_context_public()  # Make public for operations
    
    def _create_context(self):
        """Create TenSEAL CKKS context"""
        if not self.tenseal_available:
            return None
        
        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )
            context.global_scale = self.global_scale
            context.generate_galois_keys()
            return context
        except Exception as e:
            logger.error(f"Failed to create TenSEAL context: {e}")
            self.tenseal_available = False
            self._fallback = MockHEAdapter(self.enabled)
            return None
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert to numpy array"""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().astype(np.float32)
        return np.array(tensor, dtype=np.float32, copy=False)
    
    def _to_torch(self, arr: np.ndarray) -> torch.Tensor:
        """Convert to torch tensor"""
        if TORCH_AVAILABLE:
            return torch.from_numpy(arr).float()
        return arr
    
    def encrypt_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Encrypt tensor using TenSEAL CKKS with automatic chunking for large tensors"""
        if not self.enabled:
            return {"data": tensor, "encrypted": False}
        
        if not self.tenseal_available:
            return self._fallback.encrypt_tensor(tensor)
        
        arr = self._to_numpy(tensor)
        flat = arr.ravel()
        max_slots = self.poly_modulus_degree // 2  # CKKS can encrypt poly_modulus_degree/2 values
        
        # Check if we need to chunk
        if len(flat) > max_slots:
            # Chunk the tensor and encrypt each chunk
            chunks = []
            num_chunks = (len(flat) + max_slots - 1) // max_slots
            
            start_time = time.perf_counter()
            for i in range(num_chunks):
                start_idx = i * max_slots
                end_idx = min((i + 1) * max_slots, len(flat))
                chunk = flat[start_idx:end_idx].tolist()
                
                # Pad to max_slots if needed (for last chunk)
                if len(chunk) < max_slots:
                    chunk.extend([0.0] * (max_slots - len(chunk)))
                
                encrypted_chunk = ts.ckks_vector(self.context, chunk)
                serialized_chunk = encrypted_chunk.serialize()
                chunks.append(base64.b64encode(serialized_chunk).decode('ascii'))
                del encrypted_chunk, serialized_chunk  # Free memory immediately
            
            enc_time = time.perf_counter() - start_time
            gc.collect()  # Force garbage collection after chunking
            
            self.metrics["encryption_time"] += enc_time
            self.metrics["num_encryptions"] += 1
            self.metrics["total_encrypted_bytes"] += sum(len(c.encode('ascii')) for c in chunks)
            
            return {
                "encrypted": True,
                "scheme": "CKKS",
                "chunked": True,
                "data": chunks,  # List of base64-encoded chunks
                "shape": list(arr.shape),
                "dtype": "float32",
                "num_chunks": num_chunks,
                "chunk_size": max_slots,
            }
        else:
            # Single chunk - original behavior
            flat_list = flat.tolist()
            # Pad to max_slots for efficiency
            if len(flat_list) < max_slots:
                flat_list.extend([0.0] * (max_slots - len(flat_list)))
            
            try:
                start_time = time.perf_counter()
                encrypted_vector = ts.ckks_vector(self.context, flat_list)
                enc_time = time.perf_counter() - start_time
                
                # Serialize immediately and delete vector to free memory
                serialized = encrypted_vector.serialize()
                del encrypted_vector  # Free memory immediately
                gc.collect()  # Force garbage collection
                
                self.metrics["encryption_time"] += enc_time
                self.metrics["num_encryptions"] += 1
                self.metrics["total_encrypted_bytes"] += len(serialized)
                
                return {
                    "encrypted": True,
                    "scheme": "CKKS",
                    "chunked": False,
                    "data": base64.b64encode(serialized).decode('ascii'),
                    "shape": list(arr.shape),
                    "dtype": "float32",
                }
            except Exception as e:
                logger.error(f"TenSEAL encryption failed: {e}, falling back to mock")
                return self._fallback.encrypt_tensor(tensor)
    
    def decrypt_tensor(self, encrypted: Dict[str, Any], shape: Optional[Tuple] = None, dtype: str = "float32") -> Union[torch.Tensor, np.ndarray]:
        """Decrypt tensor using TenSEAL CKKS, handling chunked tensors"""
        if not encrypted.get("encrypted", False):
            return encrypted.get("data")
        
        if not self.tenseal_available or encrypted.get("scheme") != "CKKS":
            return self._fallback.decrypt_tensor(encrypted, shape, dtype)
        
        try:
            start_time = time.perf_counter()
            
            # Handle chunked encryption
            if encrypted.get("chunked", False):
                chunks = encrypted["data"]  # List of base64-encoded chunks
                decrypted_parts = []
                original_shape = tuple(encrypted.get("shape", shape or (1,)))
                total_elements = int(np.prod(original_shape))
                
                for chunk_data in chunks:
                    serialized = base64.b64decode(chunk_data)
                    encrypted_vector = ts.ckks_vector_from(self.context, serialized)
                    decrypted_chunk = encrypted_vector.decrypt()
                    decrypted_parts.extend(decrypted_chunk[:min(len(decrypted_chunk), total_elements - len(decrypted_parts))])
                    del encrypted_vector, decrypted_chunk  # Free memory immediately
                
                arr = np.array(decrypted_parts[:total_elements], dtype=np.float32)
                arr = arr.reshape(original_shape)
                del decrypted_parts  # Free intermediate list
                gc.collect()
            else:
                # Single chunk
                serialized = base64.b64decode(encrypted["data"])
                encrypted_vector = ts.ckks_vector_from(self.context, serialized)
                decrypted = encrypted_vector.decrypt()
                del encrypted_vector  # Free memory immediately
                
                arr = np.array(decrypted, dtype=np.float32)
                del decrypted  # Free decrypted list
                original_shape = tuple(encrypted.get("shape", shape or (len(arr),)))
                # Remove padding if present
                total_elements = int(np.prod(original_shape))
                arr = arr[:total_elements].reshape(original_shape)
                gc.collect()
            
            dec_time = time.perf_counter() - start_time
            
            self.metrics["decryption_time"] += dec_time
            self.metrics["num_decryptions"] += 1
            
            return self._to_torch(arr)
        except Exception as e:
            logger.error(f"TenSEAL decryption failed: {e}, falling back to mock")
            return self._fallback.decrypt_tensor(encrypted, shape, dtype)
    
    def encrypt_dict(self, tensor_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt dictionary of tensors"""
        if not self.enabled:
            return tensor_dict
        
        encrypted = {}
        for key, tensor in tensor_dict.items():
            encrypted[key] = self.encrypt_tensor(tensor)
        return encrypted
    
    def decrypt_dict(self, encrypted_dict: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Decrypt dictionary of encrypted tensors"""
        if not self.enabled:
            return encrypted_dict
        
        decrypted = {}
        for key, enc_data in encrypted_dict.items():
            if isinstance(enc_data, dict) and enc_data.get("encrypted"):
                decrypted[key] = self.decrypt_tensor(enc_data)
            else:
                decrypted[key] = enc_data
        return decrypted
    
    def aggregate_encrypted(self, encrypted_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate encrypted updates using homomorphic addition"""
        if not encrypted_list:
            return {}
        
        if not self.tenseal_available:
            return self._fallback.aggregate_encrypted(encrypted_list, weights)
        
        if weights is None:
            weights = [1.0 / len(encrypted_list)] * len(encrypted_list)
        
        all_keys = set()
        for enc_dict in encrypted_list:
            all_keys.update(enc_dict.keys())
        
        aggregated = {}
        for key in all_keys:
            try:
                encrypted_vectors = []
                valid_weights = []
                
                # Check if we're dealing with chunked tensors
                first_enc = None
                for enc_dict in encrypted_list:
                    if key in enc_dict and enc_dict[key].get("encrypted"):
                        first_enc = enc_dict[key]
                        break
                
                if first_enc is None:
                    continue
                
                is_chunked = first_enc.get("chunked", False)
                
                if is_chunked:
                    # Handle chunked aggregation - aggregate each chunk separately
                    num_chunks = first_enc.get("num_chunks", 1)
                    aggregated_chunks = []
                    
                    for chunk_idx in range(num_chunks):
                        chunk_vectors = []
                        chunk_weights = []
                        
                        for enc_dict, weight in zip(encrypted_list, weights):
                            if key in enc_dict and enc_dict[key].get("encrypted"):
                                chunk_data = enc_dict[key]["data"][chunk_idx]  # List of chunks
                                serialized = base64.b64decode(chunk_data)
                                vec = ts.ckks_vector_from(self.context, serialized)
                                chunk_vectors.append(vec)
                                chunk_weights.append(weight)
                        
                        if chunk_vectors:
                            # Aggregate this chunk
                            chunk_result = None
                            for vec, weight in zip(chunk_vectors, chunk_weights):
                                weighted = vec * weight
                                if chunk_result is None:
                                    chunk_result = weighted
                                else:
                                    chunk_result += weighted
                            
                            serialized_chunk = chunk_result.serialize()
                            aggregated_chunks.append(base64.b64encode(serialized_chunk).decode('ascii'))
                            # Aggressive cleanup
                            for v in chunk_vectors:
                                del v
                            del chunk_vectors, chunk_result, weighted
                            gc.collect()
                    
                    aggregated[key] = {
                        "encrypted": True,
                        "scheme": "CKKS",
                        "chunked": True,
                        "data": aggregated_chunks,
                        "shape": first_enc.get("shape"),
                        "dtype": "float32",
                        "num_chunks": num_chunks,
                        "chunk_size": first_enc.get("chunk_size"),
                    }
                    continue
                
                # Non-chunked aggregation (original code)
                for enc_dict, weight in zip(encrypted_list, weights):
                    if key in enc_dict and enc_dict[key].get("encrypted"):
                        serialized = base64.b64decode(enc_dict[key]["data"])
                        vec = ts.ckks_vector_from(self.context, serialized)
                        encrypted_vectors.append(vec)
                        valid_weights.append(weight)
                        del serialized  # Free serialized data
                
                if encrypted_vectors:
                    # Homomorphic weighted sum
                    result = None
                    for vec, weight in zip(encrypted_vectors, valid_weights):
                        weighted = vec * weight
                        if result is None:
                            result = weighted
                        else:
                            result += weighted
                        del weighted  # Free intermediate weighted vector
                    
                    # Serialize result and clean up memory aggressively
                    serialized_result = result.serialize()
                    # Explicitly delete all vectors to free memory
                    for vec in encrypted_vectors:
                        del vec
                    del encrypted_vectors, result, valid_weights
                    gc.collect()  # Force garbage collection
                    
                    # Check if original was chunked
                    first_enc = encrypted_list[0][key]
                    is_chunked = first_enc.get("chunked", False)
                    
                    aggregated[key] = {
                        "encrypted": True,
                        "scheme": "CKKS",
                        "chunked": is_chunked,
                        "data": base64.b64encode(serialized_result).decode('ascii'),
                        "shape": first_enc.get("shape"),
                        "dtype": "float32",
                    }
            except Exception as e:
                logger.error(f"TenSEAL aggregation failed for key {key}: {e}")
                # Fallback to mock
                if not hasattr(self, '_fallback'):
                    self._fallback = MockHEAdapter(self.enabled)
                fallback_result = self._fallback.aggregate_encrypted(
                    [{k: v for k, v in enc_dict.items() if k == key} for enc_dict in encrypted_list],
                    weights
                )
                if key in fallback_result:
                    aggregated[key] = fallback_result[key]
        
        return aggregated


class TenSEALBFVAdapter(BaseHEAdapter):
    """
    TenSEAL BFV adapter for exact homomorphic encryption.
    Good for integer operations, but less efficient for floating point.
    """
    
    def __init__(self, enabled: bool = True, poly_modulus_degree: int = 8192,
                 plain_modulus: int = 1032193, key: Optional[bytes] = None):
        super().__init__(enabled)
        
        if not TENSEAL_AVAILABLE:
            logger.warning("TenSEAL not available, falling back to MockHEAdapter")
            self._fallback = MockHEAdapter(enabled, key)
            self.tenseal_available = False
            return
        
        self.tenseal_available = True
        self.poly_modulus_degree = poly_modulus_degree
        self.plain_modulus = plain_modulus
        
        if key is None:
            key = os.urandom(32)
        self.key = key
        
        self.context = self._create_context()
        self.context.make_context_public()
    
    def _create_context(self):
        """Create TenSEAL BFV context"""
        if not self.tenseal_available:
            return None
        
        try:
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=self.poly_modulus_degree,
                plain_modulus=self.plain_modulus
            )
            context.generate_galois_keys()
            return context
        except Exception as e:
            logger.error(f"Failed to create TenSEAL BFV context: {e}")
            self.tenseal_available = False
            self._fallback = MockHEAdapter(self.enabled)
            return None
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert to numpy array, scale for BFV (integer only)"""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.array(tensor, dtype=np.float32, copy=False)
        # Scale to integers for BFV (multiply by large factor, round)
        scale_factor = 1e6  # Preserve 6 decimal places
        return (arr * scale_factor).astype(np.int64)
    
    def _from_numpy(self, arr: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        """Convert from integer numpy back to float"""
        scale_factor = 1e6
        arr_float = arr.astype(np.float32) / scale_factor
        if TORCH_AVAILABLE:
            return torch.from_numpy(arr_float).float()
        return arr_float
    
    def encrypt_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Encrypt tensor using TenSEAL BFV"""
        if not self.enabled:
            return {"data": tensor, "encrypted": False}
        
        if not self.tenseal_available:
            return self._fallback.encrypt_tensor(tensor)
        
        arr_int = self._to_numpy(tensor)
        flat = arr_int.ravel().tolist()
        
        try:
            start_time = time.perf_counter()
            encrypted_vector = ts.bfv_vector(self.context, flat)
            enc_time = time.perf_counter() - start_time
            
            serialized = encrypted_vector.serialize()
            del encrypted_vector  # Free memory immediately
            gc.collect()
            
            self.metrics["encryption_time"] += enc_time
            self.metrics["num_encryptions"] += 1
            self.metrics["total_encrypted_bytes"] += len(serialized)
            
            return {
                "encrypted": True,
                "scheme": "BFV",
                "data": base64.b64encode(serialized).decode('ascii'),
                "shape": list(arr_int.shape),
                "dtype": "int64",
                "scale_factor": 1e6,
            }
        except Exception as e:
            logger.error(f"TenSEAL BFV encryption failed: {e}, falling back to mock")
            return self._fallback.encrypt_tensor(tensor)
    
    def decrypt_tensor(self, encrypted: Dict[str, Any], shape: Optional[Tuple] = None, dtype: str = "float32") -> Union[torch.Tensor, np.ndarray]:
        """Decrypt tensor using TenSEAL BFV"""
        if not encrypted.get("encrypted", False):
            return encrypted.get("data")
        
        if not self.tenseal_available or encrypted.get("scheme") != "BFV":
            return self._fallback.decrypt_tensor(encrypted, shape, dtype)
        
        try:
            serialized = base64.b64decode(encrypted["data"])
            start_time = time.perf_counter()
            encrypted_vector = ts.bfv_vector_from(self.context, serialized)
            decrypted = encrypted_vector.decrypt()
            del encrypted_vector  # Free memory immediately
            dec_time = time.perf_counter() - start_time
            
            self.metrics["decryption_time"] += dec_time
            self.metrics["num_decryptions"] += 1
            
            arr_int = np.array(decrypted, dtype=np.int64)
            del decrypted  # Free decrypted list
            shape = tuple(encrypted.get("shape", shape or (len(arr_int),)))
            arr_int = arr_int.reshape(shape)
            gc.collect()
            
            return self._from_numpy(arr_int)
        except Exception as e:
            logger.error(f"TenSEAL BFV decryption failed: {e}, falling back to mock")
            return self._fallback.decrypt_tensor(encrypted, shape, dtype)
    
    def encrypt_dict(self, tensor_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt dictionary of tensors"""
        if not self.enabled:
            return tensor_dict
        
        encrypted = {}
        for key, tensor in tensor_dict.items():
            encrypted[key] = self.encrypt_tensor(tensor)
        return encrypted
    
    def decrypt_dict(self, encrypted_dict: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Decrypt dictionary of encrypted tensors"""
        if not self.enabled:
            return encrypted_dict
        
        decrypted = {}
        for key, enc_data in encrypted_dict.items():
            if isinstance(enc_data, dict) and enc_data.get("encrypted"):
                decrypted[key] = self.decrypt_tensor(enc_data)
            else:
                decrypted[key] = enc_data
        return decrypted
    
    def aggregate_encrypted(self, encrypted_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate encrypted updates using homomorphic addition"""
        if not encrypted_list:
            return {}
        
        if not self.tenseal_available:
            return self._fallback.aggregate_encrypted(encrypted_list, weights)
        
        if weights is None:
            weights = [1.0 / len(encrypted_list)] * len(encrypted_list)
        
        all_keys = set()
        for enc_dict in encrypted_list:
            all_keys.update(enc_dict.keys())
        
        aggregated = {}
        for key in all_keys:
            try:
                encrypted_vectors = []
                valid_weights = []
                
                for enc_dict, weight in zip(encrypted_list, weights):
                    if key in enc_dict and enc_dict[key].get("encrypted"):
                        serialized = base64.b64decode(enc_dict[key]["data"])
                        vec = ts.bfv_vector_from(self.context, serialized)
                        encrypted_vectors.append(vec)
                        # BFV requires integer weights, scale them
                        valid_weights.append(int(weight * 1e6))
                        del serialized  # Free serialized data
                
                if encrypted_vectors:
                    result = None
                    for vec, weight_int in zip(encrypted_vectors, valid_weights):
                        weighted = vec * weight_int
                        if result is None:
                            result = weighted
                        else:
                            result += weighted
                        del weighted  # Free intermediate
                    
                    # Normalize by number of clients (approximate division)
                    # In practice, you'd need to handle division differently in BFV
                    serialized_result = result.serialize()
                    # Aggressive cleanup
                    for vec in encrypted_vectors:
                        del vec
                    del encrypted_vectors, result, valid_weights
                    gc.collect()
                    
                    aggregated[key] = {
                        "encrypted": True,
                        "scheme": "BFV",
                        "data": base64.b64encode(serialized_result).decode('ascii'),
                        "shape": encrypted_list[0][key].get("shape"),
                        "dtype": "int64",
                        "scale_factor": 1e6,
                        "weight_sum": sum(valid_weights) if 'valid_weights' in locals() else 0,
                    }
            except Exception as e:
                logger.error(f"TenSEAL BFV aggregation failed for key {key}: {e}")
                if not hasattr(self, '_fallback'):
                    self._fallback = MockHEAdapter(self.enabled)
                fallback_result = self._fallback.aggregate_encrypted(
                    [{k: v for k, v in enc_dict.items() if k == key} for enc_dict in encrypted_list],
                    weights
                )
                if key in fallback_result:
                    aggregated[key] = fallback_result[key]
        
        return aggregated


class SelectiveEncryptionAdapter:
    """
    Main adapter that provides selective encryption for federated learning.
    Supports multiple HE schemes and selective layer encryption.
    
    Enhanced with features from best practices:
    - Flexible layer selection with name_filter callback
    - Metadata preservation (shape, dtype, device)
    - Option to drop plaintext when encrypted for security
    - Serialize/deserialize hooks for wire/storage format
    """
    
    def __init__(self, 
                 layers_to_encrypt: List[str] = None,
                 encryption_method: str = "mock",
                 encrypt_communication: bool = True,
                 encrypt_gradients: bool = False,
                 encrypt_updates: bool = True,
                 name_filter: Optional[Callable[[str], bool]] = None,
                 drop_plaintext_when_encrypted: bool = False,
                 **he_kwargs):
        """
        Initialize selective encryption adapter.
        
        Args:
            layers_to_encrypt: List of layer names/patterns to encrypt (e.g., ["fc", "layer1"])
            encryption_method: HE method to use ("mock", "ckks", "bfv")
            encrypt_communication: Whether to encrypt client-server communication
            encrypt_gradients: Whether to encrypt gradients during training
            encrypt_updates: Whether to encrypt model updates
            name_filter: Optional callback function for additional layer filtering (param_name -> bool)
            drop_plaintext_when_encrypted: If True, don't include plaintext when encrypted (security)
            **he_kwargs: Additional arguments for HE adapters
        """
        self.layers_to_encrypt = layers_to_encrypt or []
        self.encryption_method = encryption_method.lower()
        self.encrypt_communication = encrypt_communication
        self.encrypt_gradients = encrypt_gradients
        self.encrypt_updates = encrypt_updates
        self.name_filter = name_filter
        self.drop_plaintext_when_encrypted = drop_plaintext_when_encrypted
        
        # Initialize appropriate HE adapter
        if self.encryption_method == "ckks":
            self.he_adapter = TenSEALCKKSAdapter(enabled=True, **he_kwargs)
        elif self.encryption_method == "bfv":
            self.he_adapter = TenSEALBFVAdapter(enabled=True, **he_kwargs)
        elif self.encryption_method == "mock":
            self.he_adapter = MockHEAdapter(enabled=True, **he_kwargs)
        else:
            logger.warning(f"Unknown encryption method {encryption_method}, using mock")
            self.he_adapter = MockHEAdapter(enabled=True, **he_kwargs)
        
        self.enabled = len(self.layers_to_encrypt) > 0 or name_filter is not None
        logger.info(f"Initialized SelectiveEncryptionAdapter with method={encryption_method}, "
                   f"layers={layers_to_encrypt}, comm={encrypt_communication}")
    
    def _wrap_entry(self, 
                   data: Union[torch.Tensor, np.ndarray, Any],
                   encrypted: bool,
                   scheme: str = "none",
                   meta: Optional[dict] = None) -> dict:
        """
        Wrap data in a consistent entry format with metadata.
        Inspired by friend's cleaner structure.
        """
        meta = meta or {}
        
        # Extract tensor metadata
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            original_shape = tuple(data.shape)
            dtype = str(data.dtype)
            device = str(data.device)
            # Clone to avoid reference issues
            tensor_data = data.detach().clone()
        elif isinstance(data, np.ndarray):
            original_shape = tuple(data.shape)
            dtype = str(data.dtype)
            device = "cpu"
            tensor_data = data.copy()
        else:
            original_shape = None
            dtype = None
            device = None
            tensor_data = data
        
        entry = {
            "encrypted": encrypted,
            "scheme": scheme,  # 'none' | 'ckks' | 'bfv' | 'mock'
            "data": tensor_data,
            "original_shape": original_shape,
            "dtype": dtype,
            "device": device,
        }
        
        if meta:
            entry["meta"] = dict(meta)
        
        return entry
    
    def _should_encrypt_layer(self, layer_name: str) -> bool:
        """Check if a layer should be encrypted (with optional name_filter)"""
        if not self.enabled:
            return False
        
        # Check layer patterns
        layer_hit = False
        if not self.layers_to_encrypt:
            layer_hit = True  # Encrypt all if no specific layers specified
        else:
            for pattern in self.layers_to_encrypt:
                if pattern in layer_name or layer_name.startswith(pattern):
                    layer_hit = True
                    break
        
        # Apply additional name filter if provided
        if layer_hit and self.name_filter is not None:
            layer_hit = layer_hit and bool(self.name_filter(layer_name))
        
        return layer_hit
    
    def serialize(self, entry: dict) -> dict:
        """Serialize entry for wire/storage format. Override for custom serialization."""
        return entry
    
    def deserialize(self, entry: dict) -> dict:
        """Deserialize entry from wire/storage format. Override for custom deserialization."""
        return entry
    
    def encrypt_gradients(self, gradients: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt gradients (selective by layer) with metadata preservation"""
        if not self.enabled or not self.encrypt_gradients:
            # Return wrapped entries even when encryption is disabled
            out = {}
            for name, grad in gradients.items():
                entry = self._wrap_entry(grad, encrypted=False, scheme="none")
                out[name] = self.serialize(entry)
            return out
        
        encrypted = {}
        for layer_name, grad in gradients.items():
            if self._should_encrypt_layer(layer_name):
                # Encrypt using HE adapter
                enc_result = self.he_adapter.encrypt_tensor(grad)
                
                # Wrap in consistent format
                entry = self._wrap_entry(
                    enc_result.get("data") if isinstance(enc_result, dict) else enc_result,
                    encrypted=True,
                    scheme=self.encryption_method,
                    meta={"he_metadata": enc_result} if isinstance(enc_result, dict) else None
                )
                
                # If drop_plaintext is True, remove plaintext data
                if self.drop_plaintext_when_encrypted and isinstance(enc_result, dict):
                    entry["data"] = None  # Plaintext removed for security
                    entry["he_metadata"] = enc_result  # Keep HE metadata
                
                encrypted[layer_name] = self.serialize(entry)
                if entry["encrypted"]:
                    logger.debug(f"Encrypted grad: {layer_name} [{entry['scheme']}]")
            else:
                entry = self._wrap_entry(grad, encrypted=False, scheme="none")
                encrypted[layer_name] = self.serialize(entry)
        
        return encrypted
    
    def encrypt_update(self, update: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt model update (selective by layer) with metadata preservation"""
        if not self.enabled or not self.encrypt_updates:
            # Return wrapped entries even when encryption is disabled
            out = {}
            for name, tensor in update.items():
                entry = self._wrap_entry(tensor, encrypted=False, scheme="none")
                out[name] = self.serialize(entry)
            return out
        
        encrypted = {}
        for layer_name, tensor in update.items():
            if self._should_encrypt_layer(layer_name):
                # Encrypt using HE adapter
                enc_result = self.he_adapter.encrypt_tensor(tensor)
                
                # Wrap in consistent format
                entry = self._wrap_entry(
                    enc_result.get("data") if isinstance(enc_result, dict) else enc_result,
                    encrypted=True,
                    scheme=self.encryption_method,
                    meta={"he_metadata": enc_result} if isinstance(enc_result, dict) else None
                )
                
                # If drop_plaintext is True, remove plaintext data
                if self.drop_plaintext_when_encrypted and isinstance(enc_result, dict):
                    entry["data"] = None  # Plaintext removed for security
                    entry["he_metadata"] = enc_result  # Keep HE metadata
                
                encrypted[layer_name] = self.serialize(entry)
                if entry["encrypted"]:
                    logger.debug(f"Encrypted update: {layer_name} [{entry['scheme']}]")
            else:
                entry = self._wrap_entry(tensor, encrypted=False, scheme="none")
                encrypted[layer_name] = self.serialize(entry)
        
        return encrypted
    
    def encrypt_communication(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt data for client-server communication"""
        if not self.enabled or not self.encrypt_communication:
            return data
        
        return self.he_adapter.encrypt_dict(data)
    
    def decrypt_communication(self, encrypted_data: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Decrypt data from client-server communication.
        Handles both wrapped entry format and direct HE format.
        """
        if not self.enabled or not self.encrypt_communication:
            # If not encrypted, extract data from wrapped format if needed
            result = {}
            for key, entry in encrypted_data.items():
                if isinstance(entry, dict) and "data" in entry and not entry.get("encrypted"):
                    result[key] = entry["data"]
                else:
                    result[key] = entry
            return result
        
        # Check if wrapped format
        if encrypted_data:
            first_entry = next(iter(encrypted_data.values()))
            is_wrapped = isinstance(first_entry, dict) and "encrypted" in first_entry and "scheme" in first_entry
            
            if is_wrapped:
                # Extract HE metadata from wrapped entries
                he_dict = {}
                for key, entry in encrypted_data.items():
                    if isinstance(entry, dict) and entry.get("encrypted"):
                        if "meta" in entry and "he_metadata" in entry["meta"]:
                            he_dict[key] = entry["meta"]["he_metadata"]
                        elif isinstance(entry.get("data"), dict) and entry["data"].get("encrypted"):
                            he_dict[key] = entry["data"]
                        else:
                            # Not actually encrypted, return data as-is
                            he_dict[key] = entry.get("data")
                    else:
                        # Not encrypted, return data
                        he_dict[key] = entry.get("data") if isinstance(entry, dict) else entry
                
                # Decrypt using HE adapter
                decrypted = self.he_adapter.decrypt_dict(he_dict)
                return decrypted
            else:
                # Direct format - pass through to HE adapter
                return self.he_adapter.decrypt_dict(encrypted_data)
        
        return encrypted_data
    
    def aggregate_encrypted_updates(self, encrypted_updates: List[Dict[str, Any]], 
                                    weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Aggregate multiple encrypted updates using homomorphic operations.
        Handles both wrapped entry format and direct HE format for backward compatibility.
        """
        if not self.enabled:
            # Fallback to regular aggregation if encryption disabled
            return encrypted_updates[0] if encrypted_updates else {}
        
        # Check if we're using wrapped format (new) or direct format (old)
        if not encrypted_updates:
            return {}
        
        first_update = encrypted_updates[0]
        first_key = next(iter(first_update.keys())) if first_update else None
        
        if first_key:
            first_entry = first_update[first_key]
            # Check if it's wrapped format (has "encrypted", "scheme", "data" at top level)
            is_wrapped = isinstance(first_entry, dict) and "encrypted" in first_entry and "scheme" in first_entry
            
            if is_wrapped:
                # Extract HE metadata from wrapped entries
                he_updates = []
                for enc_dict in encrypted_updates:
                    he_dict = {}
                    for key, entry in enc_dict.items():
                        # Extract HE metadata from wrapped entry
                        if isinstance(entry, dict) and entry.get("encrypted"):
                            if "meta" in entry and "he_metadata" in entry["meta"]:
                                he_dict[key] = entry["meta"]["he_metadata"]
                            elif isinstance(entry.get("data"), dict) and entry["data"].get("encrypted"):
                                he_dict[key] = entry["data"]
                            else:
                                # Not actually encrypted, skip
                                continue
                        else:
                            # Not encrypted, skip
                            continue
                    if he_dict:
                        he_updates.append(he_dict)
                
                if he_updates:
                    # Aggregate using HE
                    agg_result = self.he_adapter.aggregate_encrypted(he_updates, weights)
                    # Return in wrapped format for consistency
                    wrapped_result = {}
                    for key, he_data in agg_result.items():
                        wrapped_result[key] = self._wrap_entry(
                            he_data.get("data") if isinstance(he_data, dict) else he_data,
                            encrypted=True,
                            scheme=self.encryption_method,
                            meta={"he_metadata": he_data} if isinstance(he_data, dict) else None
                        )
                    return wrapped_result
                else:
                    # No encrypted entries, return first update
                    return encrypted_updates[0]
            else:
                # Direct format (old) - pass through to HE adapter
                return self.he_adapter.aggregate_encrypted(encrypted_updates, weights)
        
        return {}
    
    def decrypt_and_aggregate(self, encrypted_updates_list: List[Dict[str, Any]], 
                             weights: Optional[List[float]] = None) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Decrypt and aggregate encrypted updates in one operation.
        Returns plaintext aggregated tensors.
        Inspired by friend's cleaner API.
        """
        if not encrypted_updates_list:
            return {}
        
        # Group by parameter name
        grouped: Dict[str, List[dict]] = {}
        for enc_dict in encrypted_updates_list:
            for name, entry in enc_dict.items():
                deserialized = self.deserialize(entry)
                grouped.setdefault(name, []).append(deserialized)
        
        # Aggregate per-parameter
        result: Dict[str, Union[torch.Tensor, np.ndarray]] = {}
        for name, entries in grouped.items():
            # Check if all entries are encrypted
            all_encrypted = all(e.get("encrypted", False) for e in entries)
            
            if all_encrypted:
                # Extract HE metadata and aggregate
                he_entries = []
                for e in entries:
                    # Try to get HE metadata from wrapped entry
                    if "meta" in e and "he_metadata" in e["meta"]:
                        he_entries.append(e["meta"]["he_metadata"])
                    elif isinstance(e.get("data"), dict) and e["data"].get("encrypted"):
                        # Direct HE format (backward compatibility)
                        he_entries.append(e["data"])
                    elif e.get("encrypted") and "he_metadata" in e:
                        # Alternative format
                        he_entries.append(e["he_metadata"])
                
                if he_entries:
                    # Use homomorphic aggregation
                    agg_result = self.he_adapter.aggregate_encrypted(he_entries, weights)
                    # Decrypt aggregated result - extract tensor from decrypted dict
                    decrypted_dict = self.he_adapter.decrypt_dict(agg_result)
                    # Get the tensor (should be only one key in decrypted_dict)
                    if decrypted_dict:
                        result[name] = next(iter(decrypted_dict.values()))
                    else:
                        result[name] = torch.tensor(0.0) if TORCH_AVAILABLE else np.array(0.0)
                else:
                    # Fallback - try to use data directly
                    data = entries[0].get("data")
                    if data is not None and (TORCH_AVAILABLE and isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)):
                        result[name] = data
                    else:
                        result[name] = torch.tensor(0.0) if TORCH_AVAILABLE else np.array(0.0)
            else:
                # Plaintext aggregation (mean)
                acc = None
                count = 0
                for e in entries:
                    data = e.get("data")
                    if data is not None:
                        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                            acc = data.clone() if acc is None else acc + data
                        elif isinstance(data, np.ndarray):
                            acc = data.copy() if acc is None else acc + data
                        count += 1
                
                if acc is not None and count > 0:
                    if TORCH_AVAILABLE and isinstance(acc, torch.Tensor):
                        result[name] = acc / float(count)
                    elif isinstance(acc, np.ndarray):
                        result[name] = acc / float(count)
                    else:
                        result[name] = acc
                else:
                    result[name] = torch.tensor(0.0) if TORCH_AVAILABLE else np.array(0.0)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get encryption performance metrics"""
        return {
            "adapter_method": self.encryption_method,
            "he_metrics": self.he_adapter.metrics.copy(),
            "layers_encrypted": self.layers_to_encrypt,
            "encrypt_communication": self.encrypt_communication,
            "encrypt_gradients": self.encrypt_gradients,
            "encrypt_updates": self.encrypt_updates,
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.he_adapter.reset_metrics()

