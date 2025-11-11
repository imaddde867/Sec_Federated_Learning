"""
Simple CLI entry to run a federated experiment with optional encryption.
Use from terminal or import from notebooks.

Example:
$ python run_experiment.py --rounds 1 --clients 10 --cpr 5 --enc --method mock
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from fl_env import FLConfig, FLRunner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--cpr", type=int, default=5, help="clients per round")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--enc", action="store_true", help="enable encryption")
    p.add_argument("--method", type=str, default="mock", choices=["mock", "ckks", "bfv"]) 
    p.add_argument("--layers", type=str, nargs="*", default=None)
    p.add_argument("--dp", action="store_true", help="enable DP hooks")
    p.add_argument("--clip", type=float, default=None)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FLConfig(
        num_rounds=args.rounds,
        num_clients=args.clients,
        clients_per_round=args.cpr,
        local_epochs=args.epochs,
        local_batch_size=args.batch,
        alpha=args.alpha,
        enable_encryption=args.enc,
        encryption_method=args.method,
        layers_to_encrypt=args.layers,
        enable_dp=args.dp,
        clip_norm=args.clip,
        noise_multiplier=args.noise,
    )
    runner = FLRunner(cfg)
    metrics = runner.run()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
