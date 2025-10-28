import copy
import random
import io
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import os
import hashlib
from tenseal_adapter import MockKMS, TenSEALAdapter

# -------------------- SETTINGS --------------------
number_of_clients = 1         # Number of federated clients
number_of_rounds = 1          # Number of global training rounds
local_epochs = 1              # Number of local epochs per client
use_docker = True             # Toggle Docker-style log saving
device = "cpu"                # Device: "cpu", "cuda", or "mps"
selection = "L1-2"            # Layers to encrypt
# --------------------------------------------------

# ---------- TenSEAL setup ----------
kms = MockKMS(store_dir="./.kms_fed")
adapter = TenSEALAdapter(kms=kms)
KEY_ID = "federation-key-v1"
demo_key = hashlib.sha256(b"federation_demo_key").digest()
adapter.create_context(KEY_ID, demo_key, persist=True)


# --- Dataset loading ---
def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    return train, test

train, test = load_cifar100()

# --- Dataset partition ---
def partition_dataset(dataset, num_clients, iid=True):
    n = len(dataset)
    indices = list(range(n))
    if iid:
        random.shuffle(indices)
        split = n // num_clients
        parts = [indices[i*split:(i+1)*split] for i in range(num_clients)]
        parts[-1].extend(indices[num_clients*split:])
    else:
        raise NotImplementedError("Non-iid partition not implemented yet")
    return parts

parts = partition_dataset(train, number_of_clients)

# --- TenSEAL encryption helpers ---
def client_wrap_state_dict(state_dict, selection=selection):
    return adapter.wrap_state(state_dict, key_id=KEY_ID, selection=selection)

def server_unwrap_to_state_dict(wrapped_blob, original_state_template):
    decrypted = adapter.unwrap_state(wrapped_blob, expected_key_id=KEY_ID)
    full = {}
    for k, v in original_state_template.items():
        if k in decrypted:
            full[k] = decrypted[k]
        else:
            full[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return full

# --- Local training ---
def local_train(model, dataset, device, epochs=1, batch_size=32, lr=0.01, selection=selection):
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = local_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    state = local_model.state_dict()
    wrapped_blob = client_wrap_state_dict(state, selection=selection)
    return wrapped_blob

# --- FedAvg averaging ---
def average_weights(weight_list):
    avg = copy.deepcopy(weight_list[0])
    for k in avg.keys():
        for i in range(1, len(weight_list)):
            avg[k] += weight_list[i][k]
        avg[k] = torch.div(avg[k], len(weight_list))
    return avg

# --- Log layers to file ---
def log_layers_to_file(state_dict, wrapped, round_num, client_id, file):
    """
    Logs plaintext model parameters and encrypted layer metadata (if available)
    for each training round and client.
    """
    file.write(f"\n=== Round {round_num} | Client {client_id} ===\n")

    # Log plaintext (unencrypted) model info
    file.write("Plain model layers:\n")
    for name, param in state_dict.items():
        flat = param.flatten()
        snippet = flat[:10].cpu().numpy()  # show first 10 values for preview
        file.write(f"\n{name} (shape={tuple(param.shape)}):\n")
        file.write(f"  Sample weights: {snippet}\n")

    # Log encrypted layer metrics (if provided)
    if wrapped:
        metrics = wrapped.get("metrics", {}).get("per_layer", {})
        if metrics:
            file.write("\nEncrypted layer metrics:\n")
            for layer_name, meta in metrics.items():
                enc_time = meta.get("enc_time_s", 0)
                cipher_bytes = meta.get("cipher_bytes", 0)
                file.write(
                    f" - {layer_name}: enc_time={enc_time:.4f}s, size={cipher_bytes} bytes\n"
                )
        else:
            file.write("\n[No per-layer encryption metrics found]\n")
    else:
        file.write("\n[No encryption info]\n")

    file.flush()

# --- Federated training ---
def federated_training(num_clients=number_of_clients, rounds=number_of_rounds, local_epochs=local_epochs, device=device):
    global_model = models.resnet18(num_classes=100)


    timestamp = datetime.now().strftime("%H:%M:%S_-_%d.%m.%Y")

    if use_docker:
        # Docker-style: logs folder next to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        filename = os.path.join(logs_dir, f"test-{timestamp}.txt")
    else:
        # Normal mode: save in current working directory
        filename = f"test-{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"🔐 TenSEAL Encryption enabled\n")
        f.write(f"Federated Training Log - {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        for r in range(rounds):
            print(f"\n=== Round {r+1}/{rounds} ===")
            f.write(f"=== Round {r+1}/{rounds} ===\n")
            client_states_plain = []

            for c in range(num_clients):
                print(f"Client {c+1} training...")
                f.write(f"\nClient {c+1} training...\n")
                client_dataset = Subset(train, parts[c])
                wrapped = local_train(global_model, client_dataset, device,
                                       epochs=local_epochs, batch_size=64, lr=0.01,
                                       selection=selection)
                plain_state = server_unwrap_to_state_dict(wrapped, global_model.state_dict())
                client_states_plain.append(plain_state)
                log_layers_to_file(plain_state, wrapped, r+1, c+1, f)

            averaged = average_weights(client_states_plain)
            global_model.load_state_dict(averaged)
            print(f"Round {r+1} completed.\n")
            f.write(f"\nRound {r+1} completed.\n\n")

        f.write("Training done.\n")

    print(f"✅ Log written to: {filename}")
    return global_model

# Run training
federated_training(num_clients=number_of_clients, rounds=number_of_rounds, device=device)
