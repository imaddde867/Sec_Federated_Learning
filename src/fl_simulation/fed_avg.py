#!/usr/bin/env python
# coding: utf-8

# # Federated Averaging (FedAvg) baseline on CIFAR-100
# - Dirichlet non-IID partitioning
# - Partial client participation
# - Optional heterogeneity (per-client batch size / epochs / lr)
# ## Imports and Config values

# In[1]:


import copy
import math
import time
import random
import numpy as np
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from encryption_adapter import SelectiveEncryptionAdapter


# In[ ]:


CONFIG = {
    # Federated Learning
    "num_rounds": 1,
    "num_clients": 5,
    "clients_per_round": 1,  # Partial participation
    "local_epochs": 1,
    "local_batch_size": 32,

    # Data
    "dataset": "CIFAR100",
    "data_root": "./data",
    "alpha": 0.1,  # Dirichlet concentration (lower = more non-IID)
    "num_classes": 100,  # 100 for CIFAR-100, 10 for CIFAR-10 etc..

    # Model & Training
    "model_arch": "resnet18",
    "optimizer": "SGD",
    "learning_rate": 0.001,
    "momentum": 0.0,
    "weight_decay": 0.0,

    # Capture settings
    "max_steps_to_store": 5,  # Capture at most 5 steps per client
    # Set to 0 or False to disable gradient capture, or None to store every step (not recommended).
    "return_indices": False,

    # Export
    "gradient_export_steps": 1,  # Number of gradient steps per client to persist to disk

    # Misc
    "device": "mps",
    "seed": 42,

    # Persistence
    "artifact_root": "./reports",
    "experiment_name": "fedavg_baseline",
    "save_prefix": "fedavg_metrics",
    "persist_client_payloads": False,  # Disable payload persistence to avoid large artifacts by default
    "persist_round_metrics": True,
    "persist_config_snapshot": True,

    # Privacy settings
    "enable_dp": False,           # Enable differential privacy
    "clip_norm": 1.0,             # Gradient clipping threshold
    "noise_multiplier": 1.0,      # Noise scaling factor for differential privacy
    "target_delta": 1e-5,         # Target delta parameter for DP

   # Encryption settings
    "enable_encryption": False,   # Enable encryption
    "layers_to_encrypt": ["fc"],  # Encrypt only the fully connected (fc) layer

    "enable_secure_aggregation": False,   # Enable secure aggregation

}


# In[3]:


# Reproducibility
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])


# In[4]:


# Helper Functions
def to_cpu_f32(t):
    return t.detach().to("cpu", non_blocking=True).float().clone()

def state_to_cpu_f32(sd: dict):
    return {k: to_cpu_f32(v) for k, v in sd.items()}

def param_order_and_shapes(model: torch.nn.Module):
    return [{"name": n, "shape": list(p.shape), "numel": p.numel()} 
            for n, p in model.named_parameters()]

@dataclass
class OptimCfg:
    name: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False

def build_optimizer(model, cfg: OptimCfg):
    if cfg.name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    elif cfg.name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.name}")

def class_histogram_from_loader(loader, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        y = batch[1]
        counts.index_add_(0, y.to(dtype=torch.long), torch.ones_like(y, dtype=torch.long))
    return {int(i): int(v) for i, v in enumerate(counts)}

def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def flatten_state_dict(state: dict):
    parts = []
    keys = []
    shapes = []
    dtypes = []
    for k, v in state.items():
        tensor = v.detach().cpu()
        parts.append(tensor.reshape(-1).float())
        keys.append(k)
        shapes.append(list(tensor.shape))
        dtypes.append(str(tensor.dtype))
    flat = torch.cat(parts, dim=0) if parts else torch.tensor([], dtype=torch.float32)
    template = {"keys": keys, "shapes": shapes, "dtypes": dtypes}
    return flat, template

def dict_tensor_nbytes(tensor_map: dict) -> int:
    total = 0
    for value in tensor_map.values():
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.numel()
    return int(total)

def dict_tensor_norm(tensor_map: dict) -> float:
    total = 0.0
    for value in tensor_map.values():
        if isinstance(value, torch.Tensor):
            total += float(value.float().pow(2).sum().item())
    return float(math.sqrt(total)) if total > 0.0 else 0.0

def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_client_update(
    experiment_dir: Path,
    round_idx: int,
    client_id: int,
    global_state: dict,
    local_state: dict,
    shard_size: int,
    lr: float,
    epochs: int,
    extra_meta: Optional[dict] = None,
) -> Path:
    round_dir = Path(experiment_dir) / f"round_{int(round_idx):02d}"
    ensure_dir(round_dir)

    local_flat, template = flatten_state_dict(local_state)
    global_flat, _ = flatten_state_dict(global_state)
    delta = (local_flat - global_flat).cpu()

    meta = {
        "round": int(round_idx),
        "client_id": int(client_id),
        "shard_size": int(shard_size),
        "lr": float(lr),
        "epochs": int(epochs),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "delta": delta,
        "template": template,
        "meta": meta,
    }

    path = round_dir / f"client_{client_id}.pt"
    torch.save(payload, path)
    return path


# ### Data: CIFAR-100 loaders (train/test)

# In[5]:


def load_cifar100(data_root: str = "./data"):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)
    return train, test

def _get_targets(dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise AttributeError("Dataset has no 'targets' or 'labels'.")
    return np.array(targets)


# Dirichlet non-IID split (returns dict: client_id -> list of indices)

# In[6]:


def dirichlet_noniid_indices(dataset, num_clients: int, alpha: float, 
                             min_per_client: int = 10) -> Dict[int, List[int]]:
    y = _get_targets(dataset)
    num_classes = int(y.max()) + 1
    idx_by_class = {c: np.where(y == c)[0] for c in range(num_classes)}
    for c in idx_by_class:
        np.random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        p = np.random.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
        split = np.split(idx_c, cuts)
        for i, shard in enumerate(split):
            client_indices[i].extend(shard.tolist())

    pool = list(range(len(dataset)))
    for i in range(num_clients):
        if len(client_indices[i]) < min_per_client:
            need = min_per_client - len(client_indices[i])
            extra = np.random.choice(pool, size=need, replace=False).tolist()
            client_indices[i].extend(extra)

    for i in range(num_clients):
        random.shuffle(client_indices[i])
    return {i: client_indices[i] for i in range(num_clients)}


# ### Model: ResNet18 head for CIFAR-100

# In[7]:


def _replace_bn_with_groupnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = min(32, child.num_features)
            while child.num_features % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features, affine=True)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            _replace_bn_with_groupnorm(child)


def build_model(num_classes: int = 100) -> nn.Module:
    model = models.resnet18(weights=None)  # no pretrained to avoid download in restricted envs
    # CIFAR images are 3x32x32; torchvision ResNet expects 224x224,
    # but it's fine—ResNet is fully conv except FC. It still works on 32x32.
    # Replace final FC layer to match number of classes
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    # Replace BatchNorm with GroupNorm to support batch size 1 training
    _replace_bn_with_groupnorm(model)
    return model


# In[8]:


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


# ###  Local Training with Gradient Capture

# In[9]:


@torch.no_grad()
def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def train_one_client_with_capture(
    global_model: nn.Module,
    client_loader: DataLoader,
    loss_fn: nn.Module,
    opt_cfg: OptimCfg,
    epochs: int = 1,
    device: torch.device = torch.device("cpu"),
    max_steps_to_store: int = None,
    return_indices: bool = False,
    num_classes: int = None,
    client_seed: int = None,
    clip_norm: float = None,  # Gradient clipping threshold for privacy
):
    # Create local copy of global model
    model = copy.deepcopy(global_model).to(device)
    model.train()

    # Set client-specific seed for reproducibility
    if client_seed is not None:
        torch.manual_seed(client_seed)
        random.seed(client_seed)
        np.random.seed(client_seed)

    # Save initial state to compute delta later
    global_before = clone_state(model)
    opt = build_optimizer(model, opt_cfg)

    # Storage for captured gradients and metadata
    grads_per_step_raw = []
    grads_per_step_wd = []
    batch_sizes = []
    step_losses = []
    step_batch_indices = []

    steps_stored = 0
    total_samples_seen = 0
    capture_limit = max_steps_to_store
    if isinstance(capture_limit, bool):
        capture_limit = None if capture_limit else 0
    capture_enabled = (capture_limit is None) or (isinstance(capture_limit, (int, float)) and capture_limit > 0)

    # Local training loop
    for _ in range(epochs):
        for batch in client_loader:
            # Handle batch with or without indices
            if return_indices and len(batch) == 3:
                x, y, idxs = batch
            else:
                x, y = batch[0], batch[1]
                idxs = None

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_size = x.shape[0]
            total_samples_seen += int(batch_size)

            # Forward pass and backward
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            # Gradient Clipping
            if clip_norm is not None and clip_norm > 0:
                # Compute total gradient norm (L2)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # Clip if norm exceeds threshold
                if total_norm > clip_norm:
                    clip_coef = clip_norm / (total_norm + 1e-6)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(clip_coef)

            # Store gradients (after clipping)
            should_capture = capture_enabled and (capture_limit is None or steps_stored < capture_limit)
            if should_capture:
                raw_dict = {}
                wd_dict = {}

                # Iterate through all parameters
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue

                    g = p.grad
                    # Store raw gradient (clipped if clip_norm is set)
                    raw_dict[name] = to_cpu_f32(g)

                    # Store gradient with weight decay applied
                    if opt_cfg.weight_decay and opt_cfg.weight_decay > 0:
                        wd_dict[name] = to_cpu_f32(g + opt_cfg.weight_decay * p.data)
                    else:
                        wd_dict[name] = to_cpu_f32(g)

                # Append to storage
                grads_per_step_raw.append(raw_dict)
                grads_per_step_wd.append(wd_dict)
                batch_sizes.append(int(batch_size))
                step_losses.append(float(loss.detach().item()))

                if return_indices and idxs is not None:
                    step_batch_indices.append([int(i) for i in idxs])

                steps_stored += 1

            # Update model parameters
            opt.step()

    # Get final local state
    local_after = clone_state(model)

    # Compute update delta (local_state - global_state)
    delta = OrderedDict()
    for k in local_after.keys():
        delta[k] = to_cpu_f32(local_after[k]) - to_cpu_f32(global_before[k])

    # Compute gradient statistics
    if len(grads_per_step_raw) > 0:
        first_step = grads_per_step_raw[0]
        # Per-layer gradient norms
        per_layer_norms = {k: float(v.view(-1).norm().item()) for k, v in first_step.items()}
        # Total gradient norm
        grad_norm_total = float(torch.sqrt(sum(v.pow(2).sum() for v in first_step.values())).item())
    else:
        per_layer_norms, grad_norm_total = {}, 0.0

    # Class distribution histogram
    class_dist = None
    if num_classes is not None:
        class_dist = class_histogram_from_loader(client_loader, num_classes=num_classes)

    # Package telemetry
    telemetry = {
        "per_layer_norms": per_layer_norms,
        "gradient_norm": grad_norm_total,
        "loss_history": step_losses,
        "batch_sizes": batch_sizes,
        "num_steps_captured": len(grads_per_step_raw),
        "num_samples": total_samples_seen,
        "class_distribution": class_dist,
    }

    if return_indices and len(step_batch_indices) > 0:
        telemetry["batch_indices"] = step_batch_indices

    # Return complete training result
    return {
        "local_state_after": state_to_cpu_f32(local_after),
        "delta": delta,
        "grads_per_step_raw": grads_per_step_raw,      # Raw gradients (clipped)
        "grads_per_step_wd": grads_per_step_wd,        # Gradients with weight decay
        "telemetry": telemetry,
    }


# ### FedAvg Aggregation

# In[10]:


def average_weights(weight_list, sizes):
    if not weight_list:
        raise ValueError("No client weights provided.")
    if len(weight_list) != len(sizes):
        raise ValueError("weights and sizes mismatch")

    # Compute total samples across all clients
    total = float(sum(sizes))

    # Initialize averaged weights with zeros
    avg = {k: torch.zeros_like(v) for k, v in weight_list[0].items()}

    # Weighted sum: avg = sum(weight_i * size_i / total_size)
    for wi, si in zip(weight_list, sizes):
        w = si / total  # Weight for this client
        for k in avg.keys():
            if avg[k].dtype.is_floating_point:
                # Floating point parameters: weighted average
                avg[k] += wi[k].float() * w
            else:
                # Non-floating point (e.g., batch norm buffers): copy last value
                avg[k] = wi[k].clone()

    return avg

def secure_aggregate_stub(weight_list, sizes, enabled: bool = False):
    if not enabled:
        # When disabled, fall back to standard FedAvg
        return average_weights(weight_list, sizes)

    print("Using secure aggregation (stub)")
    return average_weights(weight_list, sizes)


# ## Federated Round with Capture

# In[11]:


def run_fed_round_with_capture(
    round_num: int,
    global_model: nn.Module,
    clients: dict,
    loss_fn: nn.Module,
    opt_cfg: OptimCfg,
    local_epochs: int,
    device: torch.device,
    num_classes: int = None,
    max_steps_to_store: int = None,
    return_indices: bool = False,
    server_seed: int = None,
    client_seeds: dict = None,
    training_meta: dict = None,
    global_eval_fn = None,
    clip_norm: float = None,              # Gradient clipping for DP
    encryption_adapter = None,            # Optional encryption for gradients/updates
    client_callback: Optional[Callable[[int, int, dict, dict], None]] = None,
):
    # Set server-side seeds for reproducibility (optional)
    if server_seed is not None:
        torch.manual_seed(server_seed)
        random.seed(server_seed)
        np.random.seed(server_seed)

    # Participating clients (keys of the provided loaders)
    participating_clients = list(clients.keys())
    # Save the current global model (for deltas and logging)
    global_state_cpu = state_to_cpu_f32(global_model.state_dict())

    # Per-round containers
    client_metrics, raw_gradients, model_updates = {}, {}, {}

    # Local training on each client (with gradient capture)
    for cid, loader in clients.items():
        cseed = (client_seeds or {}).get(cid)
        result = train_one_client_with_capture(
            global_model=global_model,
            client_loader=loader,
            loss_fn=loss_fn,
            opt_cfg=opt_cfg,
            epochs=local_epochs,
            device=device,
            max_steps_to_store=max_steps_to_store,
            return_indices=return_indices,
            num_classes=num_classes,
            client_seed=cseed,
            clip_norm=clip_norm,  # per-batch gradient clipping inside client (optional)
        )

        # Optionally encrypt first-step gradients (demo / selective-layer)
        if encryption_adapter and encryption_adapter.enabled:
            encrypted_grads = encryption_adapter.encrypt_gradients(
                result["grads_per_step_raw"][0]
            )
            result["encrypted_grads"] = encrypted_grads

        # Save client delta and raw gradients (for attacks/eval)
        model_updates[cid] = result["delta"]
        raw_gradients[cid] = {
            "grads_per_step_raw": result["grads_per_step_raw"],
            "grads_per_step_wd": result["grads_per_step_wd"],
        }

        # Build client metrics (sizes, losses, norms, etc.)
        delta_bytes = dict_tensor_nbytes(result["delta"])
        delta_norm = dict_tensor_norm(result["delta"])
        tele = result["telemetry"]
        client_metrics[cid] = {
            "gradient_norm": tele["gradient_norm"],
            "per_layer_norms": tele["per_layer_norms"],
            "local_epochs": local_epochs,
            "learning_rate": opt_cfg.lr,
            "num_samples": tele["num_samples"],
            "class_distribution": tele["class_distribution"],
            "local_loss": float(tele["loss_history"][-1]) if tele["loss_history"] else None,
            "loss_history": tele["loss_history"],
            "batch_sizes": tele["batch_sizes"],
            "num_steps_captured": tele.get("num_steps_captured"),
            "update_norm": delta_norm,
            "upload_bytes": delta_bytes,
            "seed": cseed,
        }
        if return_indices and ("batch_indices" in tele):
            client_metrics[cid]["batch_indices"] = tele["batch_indices"]

        # Optional per-client save hook (e.g., write payloads to disk)
        if callable(client_callback):
            client_callback(round_num, cid, result, global_state_cpu)

        # Optionally encrypt the model update itself (selective-layer)
        if encryption_adapter and encryption_adapter.enabled:
            result["encrypted_update"] = encryption_adapter.encrypt_update(result["delta"])

    # Per-client clipping on updates (client-level DP)
    clip_C = CONFIG.get("clip_norm")
    if clip_C and clip_C > 0:
        for _cid, _delta in model_updates.items():
            _sq = 0.0
            for _v in _delta.values():
                if isinstance(_v, torch.Tensor):
                    _sq += float(_v.detach().float().pow(2).sum().item())
            _norm = math.sqrt(_sq) if _sq > 0.0 else 0.0
            if _norm > clip_C:
                _coef = clip_C / (_norm + 1e-6)
                for _k, _v in _delta.items():
                    if isinstance(_v, torch.Tensor):
                        _delta[_k] = _v * _coef

    # Aggregate updates: secure-agg stub or standard FedAvg
    agg_delta = {}
    if len(model_updates) > 0:
        keys = next(iter(model_updates.values())).keys()
        sizes = [client_metrics[cid]["num_samples"] for cid in participating_clients]
        total_samples = float(sum(sizes)) if sizes else 0.0

        use_secure_agg = CONFIG.get("enable_secure_aggregation", False)
        if use_secure_agg:
            # Stub: behaves like weighted average, but hides per-client intermediates
            agg_delta = secure_aggregate_stub(  
                list(model_updates.values()),
                sizes,
                enabled=True
            )
        else:
            # Standard FedAvg weighted average by #samples
            for k in keys:
                agg = torch.zeros_like(model_updates[participating_clients[0]][k])
                for cid, size in zip(participating_clients, sizes):
                    weight = float(size) / total_samples if total_samples > 0 else 0.0
                    agg = agg + model_updates[cid][k] * weight
                agg_delta[k] = agg

    # Add DP noise AFTER aggregation (client-level DP noise)
    if CONFIG.get("enable_dp", False):
        clip_C = CONFIG.get("clip_norm", 1.0)
        sigma = clip_C * CONFIG.get("noise_multiplier", 1.0)
        for k, v in list(agg_delta.items()):
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                agg_delta[k] = v + torch.normal(
                    mean=0.0, std=sigma, size=v.shape, device=v.device, dtype=v.dtype
                )

    # Optional global evaluation callback
    global_accuracy = None
    global_loss = None
    if callable(global_eval_fn):
        global_loss, global_accuracy = global_eval_fn(global_model)

    # Snapshot the configuration context for this round
    config_snapshot = {
        "arch": type(global_model).__name__,
        "optimizer": asdict(opt_cfg),
        "loss": type(loss_fn).__name__,
        "num_classes": num_classes,
        "param_meta": param_order_and_shapes(global_model),
        "seeds": {"server_seed": server_seed, "client_seeds": client_seeds},
        "device": str(device),
    }
    if training_meta:
        config_snapshot.update({"training_meta": training_meta})

    # Round-level stats (norms, counts)
    update_norms = {int(cid): client_metrics[cid]["update_norm"] for cid in participating_clients}
    norm_values = np.array(list(update_norms.values()), dtype=float) if update_norms else np.array([], dtype=float)
    round_stats = {
        "update_norms": update_norms,
        "server_delta_norm": dict_tensor_norm(agg_delta),
        "avg_update_norm": float(norm_values.mean()) if norm_values.size else 0.0,
        "std_update_norm": float(norm_values.std()) if norm_values.size > 1 else 0.0,
        "max_update_norm": float(norm_values.max()) if norm_values.size else 0.0,
        "min_update_norm": float(norm_values.min()) if norm_values.size else 0.0,
        "num_participating_clients": len(participating_clients),
    }

    # Communication accounting (server broadcast + client uploads)
    client_upload_details = {int(cid): int(client_metrics[cid]["upload_bytes"]) for cid in participating_clients}
    communication_bytes = {
        "server_broadcast": dict_tensor_nbytes(global_state_cpu),
        "client_upload_total": int(sum(client_upload_details.values())),
        "client_upload_per_client": client_upload_details,
    }

    # Return full round summary
    return {
        "round": int(round_num),
        "participating_clients": participating_clients,
        "client_metrics": client_metrics,
        "global_model_state": global_state_cpu,
        "global_accuracy": global_accuracy,
        "global_loss": global_loss,
        "raw_gradients": raw_gradients,
        "model_updates": model_updates,
        "server_aggregate_delta": agg_delta,
        "config_snapshot": config_snapshot,
        "round_stats": round_stats,
        "communication_bytes": communication_bytes,
    }


# ## Save Exports

# In[ ]:


import json
import pickle

def _prepare_gradients_for_export(raw_gradients, steps_to_keep):
    """Condense captured gradients to the subset required for inversion."""
    if not raw_gradients:
        return {}
    if steps_to_keep is not None and steps_to_keep <= 0:
        steps_to_keep = 1
    condensed = {}
    for cid, payload in raw_gradients.items():
        steps = payload.get("grads_per_step_raw", [])
        if steps_to_keep is not None:
            steps = steps[:steps_to_keep]
        condensed[cid] = {"grads_per_step_raw": steps}
    return condensed

def save_round_export(
    metrics_to_export,
    experiment_dir: Path,
    prefix: str = "fed_round",
    persist_json: bool = True,
    max_gradient_steps: int = 1,
):
    r = metrics_to_export["round"]
    ensure_dir(experiment_dir)
    round_dir = Path(experiment_dir) / f"round_{r:02d}"
    ensure_dir(round_dir)

    tensor_blob = {
        "round": r,
        "global_model_state": metrics_to_export["global_model_state"],
        "raw_gradients": _prepare_gradients_for_export(
            metrics_to_export.get("raw_gradients", {}), max_gradient_steps
        ),
    }
    excluded_meta_keys = set(tensor_blob.keys()) | {"model_updates", "server_aggregate_delta"}
    meta_blob = {
        k: v
        for k, v in metrics_to_export.items()
        if k not in excluded_meta_keys
    }

    tensor_path = round_dir / f"{prefix}_{r:02d}_tensors.pt"
    torch.save(tensor_blob, tensor_path)

    meta_path = None
    if persist_json:
        meta_path = round_dir / f"{prefix}_{r:02d}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_blob, f, indent=2, default=json_default)
    else:
        meta_path = round_dir / f"{prefix}_{r:02d}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(meta_blob, f)

    return {
        "round_dir": str(round_dir),
        "tensor_path": str(tensor_path),
        "meta_path": str(meta_path) if meta_path else None,
    }


# ## Final / main executionn

# In[ ]:


if __name__ == "__main__":
    print("FedAvg Training with Gradient Capture")
    print(f"Device preference: {CONFIG['device']}")

    # Resolve device (prefer CUDA, fallback to MPS or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Resolved device: {device}")

    # Display training configuration
    print(f"Rounds: {CONFIG['num_rounds']}")
    print(f"Clients: {CONFIG['num_clients']} (sampling {CONFIG['clients_per_round']}/round)")
    print(f"Local epochs: {CONFIG['local_epochs']}")
    print(f"Alpha (non-IID): {CONFIG['alpha']}")
    print("_" * 100)

    # Enable pin_memory for CUDA to speed up data transfer
    pin_memory = (device.type == "cuda")
    print("\n[1/5] Loading data...")
    train_dataset, test_dataset = load_cifar100(CONFIG["data_root"])

    # Create test loader (no shuffling needed for evaluation)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing fork issues
        pin_memory=pin_memory
    )

    print("[2/5] Partitioning data (Dirichlet non-IID)...")
    client_indices = dirichlet_noniid_indices(
        train_dataset,
        CONFIG["num_clients"],
        CONFIG["alpha"],
    )

    # Create DataLoader for each client
    client_loaders = {}
    for cid, indices in client_indices.items():
        subset = Subset(train_dataset, indices)

        # Set client-specific seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(CONFIG["seed"] + int(cid))

        client_loaders[cid] = DataLoader(
            subset,
            batch_size=CONFIG["local_batch_size"],
            shuffle=True,
            num_workers=0,  # Avoid fork issues
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=pin_memory,
            drop_last=True,
        )

    print(f"   Client data sizes: {[len(idx) for idx in client_indices.values()]}")
    print("[3/5] Building model...")
    global_model = build_model(CONFIG["num_classes"]).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Configure optimizer
    opt_cfg = OptimCfg(
        name=CONFIG["optimizer"],
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Create experiment directory
    experiment_dir = Path(CONFIG["artifact_root"]) / CONFIG["experiment_name"]
    ensure_dir(experiment_dir)

    # PRIVACY: Initialize Differential Privacy Accountant (use client sampling rate; one step per round)
    if CONFIG.get("enable_dp", False):
        from privacy_accountant import SimpleRDPAccountant
        client_sample_rate = CONFIG["clients_per_round"] / CONFIG["num_clients"]
        accountant = SimpleRDPAccountant(
            noise_multiplier=CONFIG.get("noise_multiplier", 1.0),
            sample_rate=client_sample_rate,
            target_delta=CONFIG.get("target_delta", 1e-5)
        )
        print(f"DP enabled: noise={CONFIG['noise_multiplier']}, clip={CONFIG['clip_norm']}")
    else:
        accountant = None
        print("DP disabled")

    # Save experiment configuration
    if CONFIG.get("persist_config_snapshot", True):
        config_snapshot = dict(CONFIG)
        config_snapshot.update({
            "resolved_device": str(device),
            "param_meta": param_order_and_shapes(global_model),
        })
        with open(experiment_dir / "experiment_config.json", "w") as f:
            json.dump(config_snapshot, f, indent=2, default=json_default)

    # ENCRYPTION: Initialize Selective Encryption Adapter
    if CONFIG.get("enable_encryption", False):
        from encryption_adapter import SelectiveEncryptionAdapter
        encryption_adapter = SelectiveEncryptionAdapter(
            layers_to_encrypt=CONFIG["layers_to_encrypt"]
        )
        print(f"Encryption enabled for layers: {CONFIG['layers_to_encrypt']}")
    else:
        encryption_adapter = None

    print("[4/5] Starting federated training...")

    for round_num in range(CONFIG["num_rounds"]):
        print(f"\n--- Round {round_num + 1}/{CONFIG['num_rounds']} ---")

        # Sample participating clients for this round
        participating = random.sample(
            list(client_loaders.keys()),
            CONFIG["clients_per_round"],
        )
        selected_loaders = {cid: client_loaders[cid] for cid in participating}

        round_start = time.perf_counter()
        client_artifact_paths = {}

        # CALLBACK: Define function to save client updates
        client_callback = None
        if CONFIG.get("persist_client_payloads", True):
            def _callback(r_idx, client_id, result, global_state):
                """Save client update to disk"""
                tele = result["telemetry"]

                # Determine client seed
                seed_value = None
                if client_seeds and client_id in client_seeds:
                    seed_value = client_seeds[client_id]
                elif CONFIG.get("seed") is not None:
                    seed_value = CONFIG["seed"] + int(client_id)

                # Package metadata
                extra_meta = {
                    "seed": seed_value,
                    "num_samples": int(tele["num_samples"]),
                    "num_batches": len(tele["batch_sizes"]),
                    "batch_sizes": [int(b) for b in tele["batch_sizes"]],
                    "gradient_norm": tele["gradient_norm"],
                    "num_steps_captured": tele.get("num_steps_captured"),
                }
                if tele.get("class_distribution") is not None:
                    extra_meta["class_distribution"] = tele["class_distribution"]
                if tele.get("loss_history"):
                    extra_meta["loss_history"] = tele["loss_history"]

                # Save to disk
                path = save_client_update(
                    experiment_dir=experiment_dir,
                    round_idx=r_idx,
                    client_id=client_id,
                    global_state=global_state,
                    local_state=result["local_state_after"],
                    shard_size=tele["num_samples"],
                    lr=opt_cfg.lr,
                    epochs=CONFIG["local_epochs"],
                    extra_meta=extra_meta,
                )
                client_artifact_paths[int(client_id)] = str(path.relative_to(experiment_dir))

            client_callback = _callback

        # Generate seeds for this round
        server_seed = CONFIG["seed"] + round_num if CONFIG.get("seed") is not None else None
        client_seeds = {cid: CONFIG["seed"] + round_num + int(cid) for cid in participating} if CONFIG.get("seed") is not None else None

        # RUN ONE FEDERATED ROUND
        metrics = run_fed_round_with_capture(
            round_num=round_num,
            global_model=global_model,
            clients=selected_loaders,
            loss_fn=loss_fn,
            opt_cfg=opt_cfg,
            local_epochs=CONFIG["local_epochs"],
            device=device,
            num_classes=CONFIG["num_classes"],
            max_steps_to_store=CONFIG["max_steps_to_store"],
            return_indices=CONFIG["return_indices"],
            server_seed=server_seed,
            client_seeds=client_seeds,
            training_meta={"dataset": CONFIG["dataset"], "alpha": CONFIG["alpha"]},
            global_eval_fn=lambda m: evaluate(m, test_loader, device),
            clip_norm=CONFIG.get("clip_norm"),              # Pass gradient clipping
            encryption_adapter=encryption_adapter,          # Pass encryption adapter
            client_callback=client_callback,
        )

        # Record timing
        metrics["round_wall_time_sec"] = time.perf_counter() - round_start
        metrics["client_artifacts"] = client_artifact_paths

        # Add artifact paths to client metrics
        if client_artifact_paths:
            for cid, path in client_artifact_paths.items():
                if cid in metrics["client_metrics"]:
                    metrics["client_metrics"][cid]["artifact_path"] = path

        # UPDATE GLOBAL MODEL
        current_state = global_model.state_dict()
        new_state = {}
        for k in current_state.keys():
            if k in metrics["server_aggregate_delta"]:
                # Apply aggregated delta: new = old + delta
                new_state[k] = current_state[k] + metrics["server_aggregate_delta"][k].to(device)
            else:
                new_state[k] = current_state[k]
        global_model.load_state_dict(new_state)

        # PRIVACY: Update privacy budget (one composition step per round)
        if accountant:
            accountant.step(num_steps=1)
            privacy_spent = accountant.get_privacy_spent()
            print(f"   Privacy: ε={privacy_spent['epsilon']:.2f}, δ={privacy_spent['delta']:.2e}")

        # DISPLAY ROUND RESULTS
        print(f"   Clients: {participating}")
        print(f"   Global Loss: {metrics['global_loss']:.4f}" if metrics['global_loss'] is not None else "   Global Loss: n/a")
        print(f"   Global Acc: {metrics['global_accuracy']:.4f}" if metrics['global_accuracy'] is not None else "   Global Acc: n/a")

        upload_mb = metrics["communication_bytes"]["client_upload_total"] / 1e6 if metrics["communication_bytes"]["client_upload_total"] else 0.0
        print(f"   Round time: {metrics['round_wall_time_sec']:.2f}s | Client upload: {upload_mb:.2f} MB")

        # SAVE ROUND METRICS
        export_paths = save_round_export(
            metrics,
            experiment_dir=experiment_dir,
            prefix=CONFIG["save_prefix"],
            persist_json=CONFIG.get("persist_round_metrics", True),
            max_gradient_steps=CONFIG.get("gradient_export_steps", 1),
        )

        # Helper function to convert paths to relative
        def _rel(path_str):
            if path_str is None:
                return None
            p = Path(path_str)
            try:
                return str(p.relative_to(experiment_dir))
            except ValueError:
                return str(p)

        metrics["round_artifacts"] = {
            "round_dir": _rel(export_paths["round_dir"]),
            "tensor": _rel(export_paths["tensor_path"]),
            "meta": _rel(export_paths.get("meta_path")),
        }
        print(f"   Saved artifacts under {metrics['round_artifacts']['round_dir']}")

    print("\n[5/5] Final evaluation...")
    final_loss, final_acc = evaluate(global_model, test_loader, device)
    print(f"   Final Test Loss: {final_loss:.4f}")
    print(f"   Final Test Accuracy: {final_acc:.4f}")

    print("\n" + "-" * 100)
    print("Training complete!")

