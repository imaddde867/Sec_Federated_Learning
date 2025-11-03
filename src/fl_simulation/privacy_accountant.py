import math

class SimpleRDPAccountant:

    def __init__(self, noise_multiplier: float, sample_rate: float, target_delta: float = 1e-5):
        # Noise multiplier σ used in the Gaussian mechanism
        self.noise_multiplier = float(noise_multiplier)
        # Sampling rate q ∈ (0, 1]; clamp for numerical stability
        self.sample_rate = max(1e-12, min(1.0, float(sample_rate)))
        # Target delta value δ (clamped to avoid log(0))
        self.target_delta = max(1e-12, float(target_delta))
        # Number of composition steps (in your setup, one per round)
        self.steps = 0
        # Keep last epsilon to ensure non-decreasing reporting
        self._last_eps = 0.0

    def step(self, num_steps: int = 1):
        self.steps += int(max(0, num_steps))

    def advance_round(self):
        self.step(1)

    def get_epsilon(self, target_delta: float = None) -> float:
        if target_delta is None:
            target_delta = self.target_delta
        sigma = self.noise_multiplier
        if sigma <= 0:
            return float('inf')
        if self.steps <= 0:
            return 0.0

        q = self.sample_rate

        # Very coarse α=2 Rényi DP approximation:
        # ε_RDP ≈ (q² * steps) / σ², then ε ≈ √(2 * ε_RDP)
        eps_rdp = (q ** 2) * self.steps / (sigma ** 2)
        eps = math.sqrt(max(0.0, 2.0 * eps_rdp))

        # Rough correction for δ term to prevent log divergence
        eps += math.log(1.0 / target_delta) / max(1.0, float(self.steps))

        # Ensure non-negative, monotonic epsilon progression
        eps = float(max(0.0, eps))
        if eps < self._last_eps:
            eps = self._last_eps
        self._last_eps = eps
        return eps

    def get_privacy_spent(self):
        return {
            "epsilon": self.get_epsilon(),
            "delta": self.target_delta,
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "note": "Approximate RDP (alpha=2). Use Opacus/TF-Privacy for strict accounting.",
        }