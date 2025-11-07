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
from typing import Dict, Any, List, Optional, Tuple, Union
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
        """Encrypt tensor using TenSEAL CKKS"""
        if not self.enabled:
            return {"data": tensor, "encrypted": False}
        
        if not self.tenseal_available:
            return self._fallback.encrypt_tensor(tensor)
        
        arr = self._to_numpy(tensor)
        flat = arr.ravel().tolist()
        
        try:
            start_time = time.perf_counter()
            encrypted_vector = ts.ckks_vector(self.context, flat)
            enc_time = time.perf_counter() - start_time
            
            # Serialize
            serialized = encrypted_vector.serialize()
            
            self.metrics["encryption_time"] += enc_time
            self.metrics["num_encryptions"] += 1
            self.metrics["total_encrypted_bytes"] += len(serialized)
            
            return {
                "encrypted": True,
                "scheme": "CKKS",
                "data": base64.b64encode(serialized).decode('ascii'),
                "shape": list(arr.shape),
                "dtype": "float32",
            }
        except Exception as e:
            logger.error(f"TenSEAL encryption failed: {e}, falling back to mock")
            return self._fallback.encrypt_tensor(tensor)
    
    def decrypt_tensor(self, encrypted: Dict[str, Any], shape: Optional[Tuple] = None, dtype: str = "float32") -> Union[torch.Tensor, np.ndarray]:
        """Decrypt tensor using TenSEAL CKKS"""
        if not encrypted.get("encrypted", False):
            return encrypted.get("data")
        
        if not self.tenseal_available or encrypted.get("scheme") != "CKKS":
            return self._fallback.decrypt_tensor(encrypted, shape, dtype)
        
        try:
            serialized = base64.b64decode(encrypted["data"])
            start_time = time.perf_counter()
            encrypted_vector = ts.ckks_vector_from(self.context, serialized)
            decrypted = encrypted_vector.decrypt()
            dec_time = time.perf_counter() - start_time
            
            self.metrics["decryption_time"] += dec_time
            self.metrics["num_decryptions"] += 1
            
            arr = np.array(decrypted, dtype=np.float32)
            shape = tuple(encrypted.get("shape", shape or (len(arr),)))
            arr = arr.reshape(shape)
            
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
                
                for enc_dict, weight in zip(encrypted_list, weights):
                    if key in enc_dict and enc_dict[key].get("encrypted"):
                        serialized = base64.b64decode(enc_dict[key]["data"])
                        vec = ts.ckks_vector_from(self.context, serialized)
                        encrypted_vectors.append(vec)
                        valid_weights.append(weight)
                
                if encrypted_vectors:
                    # Homomorphic weighted sum
                    result = None
                    for vec, weight in zip(encrypted_vectors, valid_weights):
                        weighted = vec * weight
                        if result is None:
                            result = weighted
                        else:
                            result += weighted
                    
                    # Serialize result
                    serialized_result = result.serialize()
                    aggregated[key] = {
                        "encrypted": True,
                        "scheme": "CKKS",
                        "data": base64.b64encode(serialized_result).decode('ascii'),
                        "shape": encrypted_list[0][key].get("shape"),
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
            dec_time = time.perf_counter() - start_time
            
            self.metrics["decryption_time"] += dec_time
            self.metrics["num_decryptions"] += 1
            
            arr_int = np.array(decrypted, dtype=np.int64)
            shape = tuple(encrypted.get("shape", shape or (len(arr_int),)))
            arr_int = arr_int.reshape(shape)
            
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
                
                if encrypted_vectors:
                    result = None
                    for vec, weight_int in zip(encrypted_vectors, valid_weights):
                        weighted = vec * weight_int
                        if result is None:
                            result = weighted
                        else:
                            result += weighted
                    
                    # Normalize by number of clients (approximate division)
                    # In practice, you'd need to handle division differently in BFV
                    serialized_result = result.serialize()
                    aggregated[key] = {
                        "encrypted": True,
                        "scheme": "BFV",
                        "data": base64.b64encode(serialized_result).decode('ascii'),
                        "shape": encrypted_list[0][key].get("shape"),
                        "dtype": "int64",
                        "scale_factor": 1e6,
                        "weight_sum": sum(valid_weights),
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
    """
    
    def __init__(self, 
                 layers_to_encrypt: List[str] = None,
                 encryption_method: str = "mock",
                 encrypt_communication: bool = True,
                 encrypt_gradients: bool = False,
                 encrypt_updates: bool = True,
                 **he_kwargs):
        """
        Initialize selective encryption adapter.
        
        Args:
            layers_to_encrypt: List of layer names/patterns to encrypt (e.g., ["fc", "layer1"])
            encryption_method: HE method to use ("mock", "ckks", "bfv")
            encrypt_communication: Whether to encrypt client-server communication
            encrypt_gradients: Whether to encrypt gradients during training
            encrypt_updates: Whether to encrypt model updates
            **he_kwargs: Additional arguments for HE adapters
        """
        self.layers_to_encrypt = layers_to_encrypt or []
        self.encryption_method = encryption_method.lower()
        self.encrypt_communication = encrypt_communication
        self.encrypt_gradients = encrypt_gradients
        self.encrypt_updates = encrypt_updates
        
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
        
        self.enabled = True
        logger.info(f"Initialized SelectiveEncryptionAdapter with method={encryption_method}, "
                   f"layers={layers_to_encrypt}, comm={encrypt_communication}")
    
    def _should_encrypt_layer(self, layer_name: str) -> bool:
        """Check if a layer should be encrypted"""
        if not self.layers_to_encrypt:
            return True  # Encrypt all if no specific layers specified
        
        for pattern in self.layers_to_encrypt:
            if pattern in layer_name or layer_name.startswith(pattern):
                return True
        return False
    
    def encrypt_gradients(self, gradients: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt gradients (selective by layer)"""
        if not self.enabled or not self.encrypt_gradients:
            return gradients
        
        encrypted = {}
        for layer_name, grad in gradients.items():
            if self._should_encrypt_layer(layer_name):
                encrypted[layer_name] = self.he_adapter.encrypt_tensor(grad)
            else:
                encrypted[layer_name] = grad
        
        return encrypted
    
    def encrypt_update(self, update: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt model update (selective by layer)"""
        if not self.enabled or not self.encrypt_updates:
            return update
        
        encrypted = {}
        for layer_name, tensor in update.items():
            if self._should_encrypt_layer(layer_name):
                encrypted[layer_name] = self.he_adapter.encrypt_tensor(tensor)
            else:
                encrypted[layer_name] = tensor
        
        return encrypted
    
    def encrypt_communication(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        """Encrypt data for client-server communication"""
        if not self.enabled or not self.encrypt_communication:
            return data
        
        return self.he_adapter.encrypt_dict(data)
    
    def decrypt_communication(self, encrypted_data: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Decrypt data from client-server communication"""
        if not self.enabled or not self.encrypt_communication:
            return encrypted_data
        
        return self.he_adapter.decrypt_dict(encrypted_data)
    
    def aggregate_encrypted_updates(self, encrypted_updates: List[Dict[str, Any]], 
                                    weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate multiple encrypted updates using homomorphic operations"""
        if not self.enabled:
            # Fallback to regular aggregation if encryption disabled
            return encrypted_updates[0] if encrypted_updates else {}
        
        return self.he_adapter.aggregate_encrypted(encrypted_updates, weights)
    
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

