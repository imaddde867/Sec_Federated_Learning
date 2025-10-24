#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def load_jsonl(p: Path):
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def scan_reports(reports_dir: Path):
    exps = []
    for exp in reports_dir.iterdir():
        if not exp.is_dir(): continue
        train = load_jsonl(exp / "train.jsonl")
        priv = {}
        pj = exp / "accountant" / "privacy.json"
        if pj.exists():
            priv = json.loads(pj.read_text())
        exps.append({"exp_id": exp.name, "train": train, "privacy": priv})
    return exps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    exps = scan_reports(Path(args.reports))
    print(json.dumps(exps, indent=2))

if __name__ == "__main__":
    main()
