#!/usr/bin/env python3
import itertools, subprocess, yaml, sys, os
from pathlib import Path

def main():
    repo = Path(__file__).resolve().parents[1]
    matrix = yaml.safe_load((repo / "configs" / "matrix.yaml").read_text())
    base_config = str(repo / "configs" / "config.yaml")

    models = matrix["models"]
    layers = matrix["layers"]
    parts = matrix["partitioning"]
    attacks = matrix["attacks"]
    seeds = matrix["seeds"]

    jobs = []
    for m, L, P, A, S in itertools.product(models, layers, parts, attacks, seeds):
        part_over = f"data.partition={P['type']}"
        if P['type'] == "dirichlet":
            part_over += f" data.dirichlet_alpha={P['alpha']}"
        overrides = f"model.name={m} encryption.layers={L} attack.name={A} seed={S} " + part_over
        jobs.append(overrides)

    for ov in jobs:
        cmd = ["python", str(repo / "scripts" / "run_experiment.py"), "--overrides"] + ov.split()
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo)

if __name__ == "__main__":
    main()
