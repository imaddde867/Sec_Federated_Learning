# Baseline Experiment (ResNet18) to reproduce previous group’s results

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    return train, test

def train_baseline():
    train, test = load_data()
    model = models.resnet18(num_classes=100)
    # TODO: Wrap in federated simulation
    # TODO: Integrate Breaching attack here later
    # TODO: Log PSNR/SSIM baseline

if __name__ == "__main__":
    train_baseline()