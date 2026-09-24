from __future__ import annotations
import numpy as np

def normalize_log_weights(log_unnormalized: np.ndarray) -> np.ndarray:
    """Stable softmax of log-weights; returns probabilities summing to 1."""
    x = np.asarray(log_unnormalized, dtype=np.float64).reshape(-1)
    c = float(np.max(x))
    w = np.exp(x - c)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        raise RuntimeError('Posterior weights degenerate.')
    return w / s

def log_prior_uniform_discrete(n: int) -> np.ndarray:
    return np.full(n, -np.log(n))
