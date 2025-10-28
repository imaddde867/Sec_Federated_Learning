import copy, torch
from torch import nn, optim
from torch.utils.data import DataLoader

def local_train(global_model, loader, device, epochs=1, batch_size=64, lr=0.01):
    """
    Performs local training on a client's DataLoader and returns the updated state dict.
    Note: now expects `loader` to already be a DataLoader, not a Dataset.
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    return model.state_dict()

def average_state_dicts(sd_list):
    avg = copy.deepcopy(sd_list[0])
    for k in avg.keys():
        for i in range(1, len(sd_list)):
            avg[k] += sd_list[i][k]
        avg[k] /= float(len(sd_list))
    return avg

@torch.no_grad()
def evaluate(model, dataset_or_loader, device, batch_size=256):
    """
    Evaluate on either a Dataset or DataLoader.
    Automatically wraps in DataLoader if a Dataset is provided.
    """
    model.eval().to(device)
    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)