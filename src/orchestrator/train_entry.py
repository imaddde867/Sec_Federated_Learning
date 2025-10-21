"""
Minimal shim showing where to insert DP + Encryption in your FL pipeline.
Replace the dummy model/gradients with your real ResNet18 + client training.
"""
import argparse, json, pathlib
import torch

from src.dp.dp import dp_sanitize_
from src.crypto.crypto import Encryptor, tensor_serialize
from src.metrics.metrics import latency_throughput

def dummy_collect_gradients(model: torch.nn.Module):
    """
    Placeholder: return per-parameter gradients. Replace with real grads.
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            g = torch.zeros_like(p.data)
        else:
            g = p.grad.data.clone()
        grads.append(g)
    return grads

def apply_defenses(grads, algorithm: str, enc_mode: str, clip: float, noise: float):
    """
    Apply (optional) DP sanitization and/or encryption to a list of gradient tensors.
    Returns (payload, enc_meta). Payload is either tensors (no enc) or list[(nonce, ct)].
    """
    if algorithm in ("dp-only", "dp-then-enc"):
        dp_sanitize_(grads, max_norm=clip, noise_multiplier=noise)

    payload = grads
    enc_meta = {}
    if algorithm in ("enc-only", "dp-then-enc"):
        enc = Encryptor.generate(mode=enc_mode)
        bufs = [tensor_serialize(g) for g in grads]
        nonce_ct = [enc.encrypt(b) for b in bufs]
        payload = nonce_ct
        enc_meta = {"mode": enc_mode, "key_len_bytes": len(enc.key)}
    return payload, enc_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="none")
    ap.add_argument("--enc_mode", default="aes-gcm-256")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    # Dummy model so this script runs independently; swap for your ResNet18 client model.
    model = torch.nn.Linear(10, 10)
    grads = [torch.randn_like(p) for p in model.parameters()]

    # Measure DP->Enc latency/throughput
    stats = latency_throughput(
        lambda: apply_defenses([g.clone() for g in grads], args.algorithm, args.enc_mode, args.clip, args.noise),
        iters=5
    )

    payload, enc_meta = apply_defenses(grads, args.algorithm, args.enc_mode, args.clip, args.noise)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "algorithm": args.algorithm,
        "enc_meta": enc_meta,
        "latency_ms": stats["latency_ms"],
        "throughput_ops_s": stats["throughput_ops_s"],
        "rounds": args.rounds,
        "clients": args.clients,
        "batch": args.batch,
        "epochs": args.epochs,
        "noise": args.noise,
        "clip": args.clip,
    }
    (out_dir / f"metrics_{args.run_id}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"status": "ok", "run_id": args.run_id, **stats}))

if __name__ == "__main__":
    main()