import hashlib, json
from pathlib import Path

def compute_exp_id(cfg_norm: dict) -> str:
    blob = json.dumps(cfg_norm, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def ensure_exp_dirs(output_dir: str, exp_id: str) -> Path:
    root = Path(output_dir) / exp_id
    for sub in ["recon", "logs", "accountant", "artifacts"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root