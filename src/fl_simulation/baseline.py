import copy
import random
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean/std
    ])
    train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    return train, test

def partition_dataset(dataset, num_clients, iid=True):
    n = len(dataset)
    indices = list(range(n))
    if iid:
        random.shuffle(indices)
        split = n // num_clients
        parts = [indices[i*split:(i+1)*split] for i in range(num_clients)]
        # last client gets remainder
        parts[-1].extend(indices[num_clients*split:])
    else:
        # non-iid: group by class (simple approach) — not implemented here
        raise NotImplementedError("Non-iid partition not implemented")
    return parts

def local_train(model, dataset, device, epochs=1, batch_size=32, lr=0.01):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    return model.state_dict()

def average_weights(weight_list):
    avg = copy.deepcopy(weight_list[0])
    for k in avg.keys():
        for i in range(1, len(weight_list)):
            avg[k] += weight_list[i][k]
        avg[k] = torch.div(avg[k], len(weight_list))
    return avg

def federated_training(num_clients=4, rounds=10, local_epochs=1, device='cpu'):
    train, test = load_cifar100()
    parts = partition_dataset(train, num_clients)
    global_model = models.resnet18(num_classes=100)
    global_state = global_model.state_dict()

    for r in range(rounds):
        selected = list(range(num_clients))  # simple: all clients
        client_states = []
        for c in selected:
            client_dataset = Subset(train, parts[c])
            local_state = local_train(global_model, client_dataset, device,
                                      epochs=local_epochs, batch_size=64, lr=0.01)
            client_states.append(local_state)
        # FedAvg
        averaged = average_weights(client_states)
        global_model.load_state_dict(averaged)
        print(f"Round {r+1}/{rounds} finished.")

    # evaluate global model
    global_model.to(device)
    global_model.eval()
    # simple accuracy:
    correct = 0
    total = 0
    loader = DataLoader(test, batch_size=128)
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = global_model(x)
            pred = out.argmax(dim=1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    print("Global test acc:", correct/total)
    return global_model