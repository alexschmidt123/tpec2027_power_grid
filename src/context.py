from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import numpy as np
from src.config import SBOEDConfig
from src.posterior import normalize_log_weights
from src.observations.likelihood import vector_gaussian_loglik
GLOBAL_SEED=77311
BELIEF_DIM=33

@dataclass
class ExperimentContext:
    """Shared context for learned, hybrid, and baseline design methods."""
    system: str
    cfg: SBOEDConfig
    horizon: int
    n_actions: int
    n_obs: int
    n_sim: int
    obs_dim: int
    obs_indices: np.ndarray
    observation_mode: str
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    robust_rule: str
    snap_up: bool
    experiment_type: str
    centres_support: np.ndarray
    U_support: np.ndarray
    log_p0: np.ndarray
    M_support: np.ndarray
    K_support: np.ndarray
    particle_features: np.ndarray
    obs_mean: float
    obs_std: float
    test_systems: list[dict[str, Any]]
    train_systems: list[dict[str, Any]]
    validation_systems: list[dict[str, Any]]
    U_test: np.ndarray
    M_test: np.ndarray
    K_test: np.ndarray
    data_dir: Path
    out_dir: Path
    oracle_tolerance: float
    fixed_sequence: list[int]
    terminal_rule_hash: str
    config_hash: str
    undercontrol_penalty: float = 10.0
    violation_penalty: float = 0.1
    control_safe_support: np.ndarray | None = None
    ocu_table_support: np.ndarray | None = None
    control_safe_test: np.ndarray | None = None
    ocu_table_test: np.ndarray | None = None
    continuous_duration_mode: bool = False
    reset_after_probe: bool = True

def resolve_oracle_tolerance(cfg: SBOEDConfig) -> float:
    oracle = dict(cfg.raw.get('oracle') or {})
    if 'tolerance' in oracle:
        return float(oracle['tolerance'])
    if 'oracle_tolerance' in cfg.raw:
        return float(cfg.raw['oracle_tolerance'])
    return 0.0001

def resolve_sigma_y(cfg: SBOEDConfig) -> float:
    obs = dict(cfg.raw.get('observation') or {})
    if 'noise_sigma' in obs:
        return float(obs['noise_sigma'])
    return float(cfg.sigma_y)

def config_sha256(cfg: SBOEDConfig) -> str:
    data = cfg.config_path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]

def update_posterior_vector(ctx: ExperimentContext, log_w: np.ndarray, action: int, y_obs: np.ndarray) -> np.ndarray:
    centres = ctx.centres_support[int(action)]
    y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    return log_w + vector_gaussian_loglik(y, centres, ctx.sigma_y)

def belief_summary(ctx: ExperimentContext, log_w: np.ndarray, observations: list[np.ndarray]) -> np.ndarray:
    w = normalize_log_weights(log_w)
    feats = np.zeros(BELIEF_DIM, dtype=np.float32)
    feats[0] = float(len(observations)) / float(ctx.horizon)
    ess = float(1.0 / np.sum(w * w))
    feats[1] = ess / float(len(w))
    feats[2] = float(np.max(w))
    feats[3] = float(np.sum(w * ctx.M_support))
    feats[4] = float(np.sqrt(max(np.sum(w * (ctx.M_support - feats[3]) ** 2), 0.0)))
    feats[5] = float(np.sum(w * ctx.K_support))
    feats[6] = float(np.sqrt(max(np.sum(w * (ctx.K_support - feats[5]) ** 2), 0.0)))
    order = np.argsort(ctx.U_support, kind='mergesort')
    u_sorted = ctx.U_support[order]
    cdf = np.cumsum(w[order])
    for i, q in enumerate((0.05, 0.25, 0.5, 0.75, 0.95)):
        idx = int(np.searchsorted(cdf, q, side='left'))
        idx = min(max(idx, 0), u_sorted.size - 1)
        feats[7 + i] = float(u_sorted[idx])
    feats[12] = 0.0  # Historical SIR dummy feature; masked by the EIG policy input.
    for i, level in enumerate(ctx.u_grid[:16]):
        feats[13 + i] = float(np.sum(w[np.isclose(ctx.U_support, level)]))
    if observations:
        y = np.concatenate([np.asarray(o, dtype=np.float64).reshape(-1) for o in observations])
        feats[29] = float(y.mean())
        feats[30] = float(y.std() if y.size > 1 else 0.0)
        feats[31] = float(y.min())
        feats[32] = float(y.max())
    return feats

def experiment_out_dir(cfg, root, experiment_type="eig_based", create_new=False):
    raise ValueError("Pass an explicit output directory to build_sir_context")
