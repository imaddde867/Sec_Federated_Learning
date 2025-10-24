import numpy as np
from torch.utils.data import Subset
from collections import defaultdict
import random

def iid_partitions(dataset, num_clients):
    n = len(dataset)
    indices = np.random.permutation(n)
    parts = np.array_split(indices, num_clients)
    return [Subset(dataset, p.tolist()) for p in parts]

def dirichlet_partitions(dataset, num_clients, alpha=0.5):
    # assumes dataset has targets/labels in dataset.targets
    labels = np.array(dataset.targets)
    n_classes = labels.max() + 1
    idx_by_class = [np.where(labels==c)[0] for c in range(n_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx_c = idx_by_class[c]
        np.random.shuffle(idx_c)
        # draw proportions for this class
        proportions = np.random.dirichlet(alpha=np.ones(num_clients)*alpha)
        # split indices accordingly
        splits = (np.cumsum(proportions)*len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, splits)
        for i, p in enumerate(parts):
            client_indices[i].extend(p.tolist())

    # shuffle each client's indices
    for i in range(num_clients):
        random.shuffle(client_indices[i])
    return [Subset(dataset, client_indices[i]) for i in range(num_clients)]

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_federated_dataloaders(cfg_data):
    """
    Builds federated client dataloaders and a global test loader.
    cfg_data: dict with keys like 'partition', 'num_clients', 'dirichlet_alpha', 'batch_size'
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    num_clients = cfg_data.get("num_clients", 5)
    partition = cfg_data.get("partition", "iid")
    alpha = cfg_data.get("dirichlet_alpha", 0.5)

    # Split dataset indices
    if partition == "iid":
        client_parts = iid_partitions(dataset, num_clients)
    elif partition == "dirichlet":
        client_parts = dirichlet_partitions(dataset, num_clients, alpha=alpha)
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    # Wrap each subset in a DataLoader
    client_loaders = {
        cid: DataLoader(subset, batch_size=cfg_data.get("batch_size", 64), shuffle=True)
        for cid, subset in enumerate(client_parts)
    }

    test_loader = DataLoader(testset, batch_size=256, shuffle=False)
    return client_loaders, test_loader