"""
Minimal test script: loads one val image using LOC_val_solution.csv,
runs a forward pass through ResNet152, computes gradients, and calls breaching attack.
"""

from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import breaching  # assumes installed

# ---------- CONFIG (change if needed) ----------
IMAGENET_ROOT = Path("Data")   # change it to your own path
LOC_VAL_CSV = IMAGENET_ROOT / "LOC_val_solution.csv"
LOC_TRAIN_CSV = IMAGENET_ROOT / "LOC_train_solution.csv"   # optional
# VAL_DIR = IMAGENET_ROOT / "ILSVRC2012_img_val"             # optional if you want to load actual images
IMG_INDEX = 1200
DEVICE = torch.device("cpu")
DTYPE = torch.float
# ------------------------------------------------

# transforms (match repo)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transforms_pipeline = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# helper: build wnid->idx if synset mapping present
def build_wnid2idx(synfile: Path):
    if not synfile.exists():
        return None
    wnids = []
    for line in synfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        # first token usually wnid like 'n01440764'
        if toks[0].startswith("n"):
            wnids.append(toks[0])
        else:
            maybe = next((t for t in toks if t.startswith("n")), None)
            if maybe:
                wnids.append(maybe)
    if len(wnids) >= 1000:
        return {wn: i for i, wn in enumerate(wnids)}
    return None

wnid2idx = None

def label_to_index(raw):
    # numeric 1..1000 -> convert to 0..999
    try:
        n = int(raw)
        if 1 <= n <= 1000:
            return n - 1
        return n
    except Exception:
        if isinstance(raw, str) and raw.startswith("n") and wnid2idx:
            return wnid2idx.get(raw, -1)
        return -1

# --- load CSV and pick one row ---
df = pd.read_csv(LOC_VAL_CSV, header=None)
if df.shape[1] >= 2:
    df = df.iloc[:, :2]
    df.columns = ["image", "label"]
else:
    raise RuntimeError(f"{LOC_VAL_CSV} not in expected format (need image,label columns)")

row = df.iloc[IMG_INDEX]
fname = str(row["image"])
raw_label = row["label"]
label_idx = label_to_index(raw_label)
if label_idx < 0:
    print("Warning: label mapping failed; label index is -1")

# Create a dummy image tensor (C,H,W), normalized, and a Long label on the correct device.
# Use DEVICE and DTYPE from your CONFIG

# create random image tensor 
datapoint = torch.rand(3, 224, 224, device=DEVICE, dtype=DTYPE)

# build mean/std on same device/dtype and normalize
mean_tensor = torch.tensor(mean, device=DEVICE, dtype=DTYPE).view(3, 1, 1)
std_tensor = torch.tensor(std, device=DEVICE, dtype=DTYPE).view(3, 1, 1)
datapoint = (datapoint - mean_tensor) / std_tensor

# ensure label is a Long tensor and put it on the device (do NOT change its dtype to float)
if label_idx < 0:
    print("Warning: label mapping failed; using label 0 as placeholder")
    label_idx = 0

labels = torch.tensor([label_idx], dtype=torch.long, device=DEVICE)


# ---------- model, loss, move to device/dtype ----------
setup = dict(device=DEVICE, dtype=DTYPE)
model = torchvision.models.resnet152(pretrained=True)
model = model.to(**setup)
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()

# move inputs to device/dtype
datapoint = datapoint.to(**setup)
labels = labels.to(device=DEVICE)

# ---- compute loss and gradients ----
# add batch dim
outputs = model(datapoint[None, ...])
loss = loss_fn(outputs, labels)

# compute gradients w.r.t. model parameters
grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=False, create_graph=False)

# ---- prepare payloads for breaching API (matches your original structure) ----
# metadata object from your original code (kept minimal)
data_cfg_default = type("cfg", (), {})()
data_cfg_default.modality = "vision"
data_cfg_default.size = (1_281_167,)
data_cfg_default.classes = 1000
data_cfg_default.shape = (3, 224, 224)
data_cfg_default.normalize = True
data_cfg_default.mean = mean
data_cfg_default.std = std

server_payload = [
    dict(
        parameters=[p for p in model.parameters()],
        buffers=[b for b in model.buffers()],
        metadata=data_cfg_default,
    )
]
shared_data = [
    dict(
        gradients=grads,
        buffers=None,
        metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None),
    )
]

# ---- run attack ----
cfg_attack = breaching.get_attack_config("invertinggradients")
attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)

print("Attack stats:", stats)
# If the attack returned an image tensor, you can inspect or save it depending on breaching API output.