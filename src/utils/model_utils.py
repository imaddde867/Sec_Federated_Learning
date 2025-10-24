import torch
from torchvision import models as tvm
from src.models.smallcnn import SmallCNN

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vgg16":
        m = tvm.vgg16(weights=None)
        m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name in ("smallcnn","small-cnn","small_cnn"):
        return SmallCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model {name}")

def get_model(name: str, num_classes: int = 10):
    """
    Thin wrapper for backward compatibility with pipeline imports.
    """
    return build_model(name, num_classes)