This repository was augmented with scaffolding to run structured experiments for DP-based gradient encryption in federated learning.

Highlights:
- `configs/config.yaml` and `configs/matrix.yaml` (Hydra‑style) for reproducible sweeps.
- `scripts/run_experiment.py` CLI runner that generates a deterministic `exp_id` (SHA256 of normalized config), snapshots configs, and logs to JSONL.
- Foldered outputs under `reports/<exp_id>/` including `recon/`, `accountant/`, `logs/`.
- Stubs for DP selective encryption and attacks to be wired into your training code:
  - `src/encryption/dp_selective.py`
  - `src/attacks/{yin_attack.py, random_attack.py}`
  - `src/orchestrator/pipeline.py`
- Analysis utilities:
  - `src/analysis/analyze_reports.py` to collect artifacts and metrics
  - `src/analysis/weekly_summary.py` for quick weekly overviews
- CI smoke test in `.github/workflows/ci-smoke.yaml`

Next steps:
1) Replace the simulator logic in `src/orchestrator/pipeline.py` with your actual FL training loop and call the encryptor on gradients before upload.
2) Implement Yin et al. attack and privacy metrics using your image data; save reconstructions into `reports/<exp_id>/recon/`.
3) Optionally wire MLflow in `src/utils/logging_utils.py` and `scripts/run_experiment.py`.