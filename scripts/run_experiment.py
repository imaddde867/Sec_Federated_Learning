#!/usr/bin/env python3
import os, json, argparse, sys
from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[1]))
from pathlib import Path

import yaml

from src.utils.logging_utils import JsonlLogger, get_git_commit
from src.utils.config_utils import load_config, normalize_config
from src.utils.exp_utils import compute_exp_id, ensure_exp_dirs
from src.orchestrator.pipeline import run_training_rounds

def main():
    parser = argparse.ArgumentParser(description="Federated DP-Encryption experiment runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--overrides", type=str, nargs="*", default=[],
                        help="YAML-like overrides e.g. model.name=resnet18 encryption.layers=L1-2 seed=2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    cfg["commit"] = cfg.get("commit") or os.environ.get("COMMIT_SHA") or get_git_commit(default="notgit")
    cfg_norm = normalize_config(cfg)
    exp_id = compute_exp_id(cfg_norm)
    cfg.setdefault("orchestration", {})["exp_id"] = exp_id

    exp_root = ensure_exp_dirs(cfg["project"]["output_dir"], exp_id)
    (exp_root / "config_snapshot.json").write_text(json.dumps(cfg_norm, indent=2))

    logger = JsonlLogger(exp_root / "train.jsonl", extra={"exp_id": exp_id, "commit": cfg["commit"], "seed": cfg["seed"]})
    logger.info("exp_start", cfg=cfg_norm)

    artifacts = run_training_rounds(cfg, exp_root, logger)

    logger.info("exp_end", artifacts=artifacts)
    print(f"OK exp_id={exp_id} output={exp_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
