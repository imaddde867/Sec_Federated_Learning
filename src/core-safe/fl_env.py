"""
Federated Learning environment with optional homomorphic encryption and DP-ready hooks.

- Minimal, modular, redundancy-free orchestration to run FedAvg experiments.
- Encryption is pluggable via SelectiveEncryptionAdapter (mock/ckks/bfv).
- Differential Privacy hooks are provided but off by default (clip + noise stubs).

Usage (from a notebook):

from fl_env import FLConfig, FLRunner
cfg = FLConfig()  # customize as needed
runner = FLRunner(cfg)
metrics = runner.run()

"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Local imports (support both package and script imports)
try:
    from .encryption_adapter import SelectiveEncryptionAdapter
    from .dp_utils import clip_gradients_inplace, add_noise_to_update
except Exception:
    from encryption_adapter import SelectiveEncryptionAdapter
    from dp_utils import clip_gradients_inplace, add_noise_to_update

# ----------------------------
# Config
# ----------------------------

@dataclass
class FLConfig:
    # FL
    num_rounds: int = 1
    num_clients: int = 10
    clients_per_round: int = 5
    local_epochs: int = 1
    local_batch_size: int = 32

    # Data
    dataset: str = "CIFAR100"
    data_root: str = "./data"
    alpha: float = 0.1  # Dirichlet concentration
    num_classes: int = 100

    # Training
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False

    # Device/Seed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Persistence / exports
    artifact_root: str = "./reports"
    experiment_name: str = "fl_encryption"
    export_client_updates: bool = False  # if True, save per-client deltas per round
    export_dirname: str = "client_updates"  # under artifact_root/experiment_name

    # DP hooks
    enable_dp: bool = False
    clip_norm: Optional[float] = None
    noise_multiplier: float = 0.0  # sigma for Gaussian noise; 0 -> no noise

    # Encryption
    enable_encryption: bool = False
    encryption_method: str = "mock"  # mock|ckks|bfv
    layers_to_encrypt: Optional[List[str]] = None  # None -> all
    encrypt_gradients: bool = False
    encrypt_updates: bool = True
    encrypt_communication: bool = False

    # TenSEAL params (used when method is ckks/bfv)
    ckks_poly_modulus_degree: int = 8192
    ckks_coeff_mod_bit_sizes: List[int] = (60, 40, 40, 60)
    ckks_global_scale: float = 2 ** 40

    bfv_poly_modulus_degree: int = 8192
    bfv_plain_modulus: int = 1032193


# ----------------------------
# Utilities
# ----------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: FLConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def load_dataset(name: str, root: str):
    if name.upper() == "CIFAR100":
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
        train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
        return train, test
    raise ValueError(f"Unsupported dataset: {name}")


def _get_targets(dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise AttributeError("Dataset has no 'targets' or 'labels'.")
    return np.array(targets)


def dirichlet_split(dataset, num_clients: int, alpha: float, min_per_client: int = 10) -> Dict[int, List[int]]:
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


def build_client_loaders(train_ds, client_to_indices: Dict[int, List[int]], batch_size: int) -> Dict[int, DataLoader]:
    g = torch.Generator()
    loaders = {}
    for cid, idxs in client_to_indices.items():
        subset = Subset(train_ds, idxs)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2, generator=g)
    return loaders


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


@torch.no_grad()
def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ----------------------------
# Core Runner
# ----------------------------

class FLRunner:
    def __init__(self, cfg: FLConfig):
        self.cfg = cfg
        set_all_seeds(cfg.seed)
        self.device = torch.device(cfg.device)

        # Data
        train_ds, test_ds = load_dataset(cfg.dataset, cfg.data_root)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
        mapping = dirichlet_split(train_ds, cfg.num_clients, cfg.alpha)
        self.client_loaders = build_client_loaders(train_ds, mapping, cfg.local_batch_size)
        self.client_sizes = {cid: len(mapping[cid]) for cid in mapping}

        # Model
        self.global_model = build_model(cfg.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Encryption adapter
        he_kwargs = {}
        if cfg.encryption_method == "ckks":
            he_kwargs = dict(
                poly_modulus_degree=cfg.ckks_poly_modulus_degree,
                coeff_mod_bit_sizes=list(cfg.ckks_coeff_mod_bit_sizes),
                global_scale=cfg.ckks_global_scale,
            )
        elif cfg.encryption_method == "bfv":
            he_kwargs = dict(
                poly_modulus_degree=cfg.bfv_poly_modulus_degree,
                plain_modulus=cfg.bfv_plain_modulus,
            )
        self.enc = SelectiveEncryptionAdapter(
            layers_to_encrypt=cfg.layers_to_encrypt or [],
            encryption_method=cfg.encryption_method,
            encrypt_communication=cfg.encrypt_communication,
            encrypt_gradients=cfg.encrypt_gradients,
            encrypt_updates=cfg.encrypt_updates,
            drop_plaintext_when_encrypted=False,
            **he_kwargs,
        ) if cfg.enable_encryption else None

    def _train_one_client(self, cid: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        model = build_model(cfg.num_classes).to(self.device)
        model.load_state_dict(self.global_model.state_dict())
        model.train()

        opt = build_optimizer(model, cfg)
        for _ in range(cfg.local_epochs):
            for x, y in self.client_loaders[cid]:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = self.criterion(logits, y)
                loss.backward()

                # DP hooks: clip grads in-place
                if cfg.enable_dp and cfg.clip_norm and cfg.clip_norm > 0:
                    clip_gradients_inplace(model.parameters(), max_norm=cfg.clip_norm)

                opt.step()

        # Delta = local - global
        local_state = clone_state(model)
        global_state = clone_state(self.global_model)
        delta = {k: (local_state[k] - global_state[k]).detach().to("cpu").float() for k in local_state}

        # DP hooks: add noise to delta
        if cfg.enable_dp and cfg.noise_multiplier and cfg.noise_multiplier > 0:
            delta = add_noise_to_update(delta, sigma=cfg.noise_multiplier)
        return delta

    def _aggregate(self, deltas: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
        # Weighted average of deltas
        total = float(sum(sizes))
        keys = deltas[0].keys()
        agg = {k: torch.zeros_like(next(iter(deltas))[k]) for k in keys}
        for d, n in zip(deltas, sizes):
            w = n / total
            for k in keys:
                agg[k] += d[k] * w
        return agg

    def _aggregate_encrypted(self, updates_wrapped: List[Dict[str, dict]], sizes: List[int]) -> Dict[str, torch.Tensor]:
        # weights normalized to 1.0
        total = float(sum(sizes))
        weights = [n / total for n in sizes]
        # adapter returns plaintext aggregated tensors
        assert self.enc is not None
        aggregated = self.enc.decrypt_and_aggregate(updates_wrapped, weights)
        # ensure tensors are float CPU tensors
        for k, v in aggregated.items():
            if isinstance(v, torch.Tensor):
                aggregated[k] = v.detach().cpu().float()
        return aggregated

    def run(self):
        cfg = self.cfg
        metrics = {
            "rounds": [],
            "test_acc": [],
            "test_loss": [],
            "encryption": cfg.enable_encryption,
            "encryption_method": cfg.encryption_method if cfg.enable_encryption else None,
            "dp_enabled": cfg.enable_dp,
        }
        export_root: Optional[Path] = None
        if cfg.export_client_updates:
            export_root = Path(cfg.artifact_root) / cfg.experiment_name / cfg.export_dirname
            export_root.mkdir(parents=True, exist_ok=True)

        for r in range(cfg.num_rounds):
            t0 = time.time()
            # Sample clients
            all_ids = list(self.client_loaders.keys())
            selected = random.sample(all_ids, k=min(cfg.clients_per_round, len(all_ids)))

            # Local updates
            deltas: List[Dict[str, torch.Tensor]] = []
            sizes: List[int] = []
            for cid in selected:
                d = self._train_one_client(cid)
                deltas.append(d)
                sizes.append(self.client_sizes[cid])
                # Optional export for attack experiments (plaintext deltas)
                if export_root is not None:
                    torch.save({"client_id": cid, "delta": d}, export_root / f"round{r:03d}_client{cid:03d}.pt")

            # Encrypt (optional) and aggregate
            if cfg.enable_encryption and cfg.encrypt_updates:
                wrapped = [self.enc.encrypt_update(d) for d in deltas]
                agg_delta = self._aggregate_encrypted(wrapped, sizes)
                he_metrics = self.enc.get_metrics()["he_metrics"]
            else:
                agg_delta = self._aggregate(deltas, sizes)
                he_metrics = None

            # Apply update: global = global + agg_delta
            with torch.no_grad():
                state = self.global_model.state_dict()
                for k in state.keys():
                    state[k] = state[k] + agg_delta[k].to(state[k].dtype)
                self.global_model.load_state_dict(state)

            # Eval
            loss, acc = evaluate(self.global_model, self.test_loader, self.device)
            round_time = time.time() - t0
            metrics["rounds"].append({
                "round": r,
                "time_sec": round_time,
                "he_metrics": he_metrics,
            })
            metrics["test_acc"].append(float(acc))
            metrics["test_loss"].append(float(loss))

        return metrics
