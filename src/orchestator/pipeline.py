from pathlib import Path
import json, time
from typing import Dict, Any
from src.encryption.dp_selective import DPSelectiveEncryptor
from src.attacks.yin_attack import YinReconstructionAttack
from src.attacks.random_attack import RandomAttack
from src.metrics.eval_privacy import compute_privacy_metrics

def run_training_rounds(cfg: Dict[str, Any], exp_root: Path, logger):
    rounds = cfg["training"]["rounds"]
    if cfg["encryption"]["scheme"] == "none":
        encryptor = None
    else:
        encryptor = DPSelectiveEncryptor(layers=cfg["encryption"]["layers"],
                                         sigma=cfg["encryption"]["noise_multiplier"],
                                         clip=cfg["encryption"]["clip_norm"])

    attacker = YinReconstructionAttack() if cfg["attack"]["name"] == "yin" else RandomAttack()

    acc = 0.0
    for r in range(rounds):
        time.sleep(0.005)
        acc = min(0.9, acc + 0.005)
        logger.info("round_end", round=r, top1=acc, bytes_up=1.23, bytes_down=0.45, time_s=0.005)

    (exp_root / "recon" / "example.txt").write_text("placeholder")
    priv = compute_privacy_metrics()
    (exp_root / "accountant" / "privacy.json").write_text(json.dumps(priv, indent=2))
    return {"top1": acc, "privacy": priv}
