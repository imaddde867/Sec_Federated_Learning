#!/usr/bin/env python3
import argparse, json, datetime
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=args.days)
    out = {"cutoff": cutoff.isoformat()+"Z", "n_experiments": 0, "best": {}}
    out["n_experiments"] = sum(1 for _ in Path(args.reports).iterdir())
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
