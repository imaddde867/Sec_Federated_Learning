from __future__ import annotations
import itertools, yaml, subprocess, os, json, time, uuid, pathlib, sys
from datetime import datetime

def cartesian_product(dict_lists):
    keys = list(dict_lists.keys())
    vals = list(dict_lists.values())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    matrix_path = root / "experiments" / "experiment_matrix.yaml"
    matrix = yaml.safe_load(open(matrix_path))
    out_dir = root / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for cfg in cartesian_product(matrix):
        run_id = str(uuid.uuid4())[:8]
        cfg["run_id"] = run_id
        cfg_path = out_dir / f"cfg_{run_id}.json"
        json.dump(cfg, open(cfg_path, "w"), indent=2)

        cmd = [
            sys.executable, "-u", "-m", "src.orchestrator.train_entry",
            f"--algorithm={cfg['algorithms']}",
            f"--enc_mode={cfg['encryption_modes']}",
            f"--noise={cfg['noise_multiplier']}",
            f"--clip={cfg['clip_norm']}",
            f"--clients={cfg['clients_per_round']}",
            f"--epochs={cfg['local_epochs']}",
            f"--batch={cfg['batch_size']}",
            f"--rounds={cfg['rounds']}",
            f"--run_id={run_id}",
            f"--out_dir={out_dir.as_posix()}",
        ]

        print("Launching:", " ".join(cmd))
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            status = "ok"; stdout = res.stdout; stderr = res.stderr
        except subprocess.CalledProcessError as e:
            status = "error"; stdout = e.stdout or ""; stderr = e.stderr or str(e)

        dt = time.perf_counter() - t0
        (out_dir / f"log_{run_id}.out").write_text(stdout)
        (out_dir / f"log_{run_id}.err").write_text(stderr)
        meta = {"cfg_path": cfg_path.as_posix(), "status": status, "seconds": dt}
        (out_dir / f"meta_{run_id}.json").write_text(json.dumps(meta, indent=2))
        runs.append(meta)

    (out_dir / "summary.json").write_text(json.dumps({"runs": runs}, indent=2))
    print(f"Wrote results to {out_dir}")

if __name__ == "__main__":
    main()