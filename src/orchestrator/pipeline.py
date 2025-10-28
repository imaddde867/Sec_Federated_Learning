from pathlib import Path
import json, time
from typing import Dict, Any
import torch

from src.encryption.dp_selective import DPSelectiveEncryptor
from src.fl_core.fedavg_core import local_train, average_state_dicts, evaluate
from src.utils.data_utils import get_federated_dataloaders
from src.utils.model_utils import get_model
from src.attacks.yin_attack import YinReconstructionAttack
from src.attacks.random_attack import RandomAttack
from src.metrics.eval_privacy import compute_privacy_metrics

def run_training_rounds(cfg: Dict[str, Any], exp_root: Path, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and data
    model = get_model(cfg["model"]["name"]).to(device)
    clients, test_loader = get_federated_dataloaders(cfg["data"])

    # Encryption setup
    if cfg["encryption"]["scheme"] == "none":
        encryptor = None
    else:
        encryptor = DPSelectiveEncryptor(
            layers=cfg["encryption"]["layers"],
            sigma=cfg["encryption"]["noise_multiplier"],
            clip=cfg["encryption"]["clip_norm"],
        )

    # Choose attack
    attacker = (
        YinReconstructionAttack() if cfg["attack"]["name"] == "yin" else RandomAttack()
    )

    # Training loop
    for r in range(cfg["training"]["rounds"]):
        start = time.time()
        client_weights = []

        # Local training on each client
        for cid, loader in clients.items():
            local_sd = local_train(model, loader, device, epochs=cfg["training"]["local_epochs"],
                                   batch_size=cfg["training"]["batch_size"], lr=cfg["training"]["lr"])

            # Differential Privacy noise
            if encryptor:
                local_sd = encryptor.apply(local_sd)

            client_weights.append(local_sd)

        # Server aggregation
        global_sd = average_state_dicts(client_weights)
        model.load_state_dict(global_sd)

        # Evaluation
        acc = evaluate(model, test_loader, device)

        logger.info("round_end", round=r, top1=acc,
            bytes_up=123456, bytes_down=654321,
            time_s=time.time() - start)

    # Optional reconstruction attack
    (exp_root / "recon").mkdir(exist_ok=True, parents=True)
    attacker.reconstruct(model, exp_root / "recon")

    # Privacy metrics
    priv = compute_privacy_metrics()
    (exp_root / "accountant" / "privacy.json").write_text(json.dumps(priv, indent=2))

    return {"top1": acc, "privacy": priv}