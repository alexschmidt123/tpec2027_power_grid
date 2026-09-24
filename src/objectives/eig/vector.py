"""Vector-observation EIG training/evaluation on the physical delta-f bank."""
from __future__ import annotations
import csv
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from src.context import GLOBAL_SEED, ExperimentContext, update_posterior_vector
from src.posterior import normalize_log_weights
from src.policies.rl_sboed import AdaptiveExperimentPolicy, PolicyConfig, StateValueCritic
from src.layout import model_dir
from src.domains.sir.design import chronological_feasible

def _eig_feasible(ctx: ExperimentContext, actions: list[int], *, remaining_steps: int | None=None) -> np.ndarray:
    """Return actions allowed by the domain's sequential-design constraints.

    SIR measurement times are chronological. Power-grid probe durations are
    independent experimental designs: they may be selected in any order, but
    an already-used duration is masked.

    ``remaining_steps`` counts actions still to choose *including* the current
    step (defaults to ``horizon - len(actions)``).
    """
    chronological = str(getattr(ctx, 'observation_mode', '')).startswith('sir_')
    if chronological:
        rem = int(remaining_steps) if remaining_steps is not None else max(int(ctx.horizon) - len(actions), 0)
        return chronological_feasible(ctx.n_actions, actions, remaining_steps=rem)
    return np.asarray([a for a in range(ctx.n_actions) if a not in set(actions)], dtype=int)
METHODS = ('dad_eig', 'rl_sboed_eig', 'step_dad', 'myopic_delta_h', 'random', 'fixed_open_loop')

def _soft_bc_loss(logits: torch.Tensor, scores: np.ndarray, feasible: np.ndarray, *, temperature: float) -> torch.Tensor:
    """EIG policy utilities; historical tensor shapes are retained for saved checkpoints."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    feasible_scores = np.asarray(scores, dtype=np.float64)[feasible]
    if feasible_scores.size == 0 or not np.all(np.isfinite(feasible_scores)):
        raise RuntimeError(f'Non-finite EIG scores reached behavioral cloning for feasible actions {np.asarray(feasible, dtype=int).tolist()}')
    temp = max(float(temperature), 0.001)
    feas = torch.as_tensor(np.asarray(feasible, dtype=int), device=logits.device)
    feasible_logits = logits[:, feas]
    if bool((feasible_logits <= -100000000.0).any()):
        raise RuntimeError('EIG behavioral-cloning feasible actions disagree with the policy feasible-action mask; refusing to optimize invalid targets.')
    raw = torch.as_tensor(feasible_scores, dtype=torch.float32, device=logits.device)
    target = torch.softmax((raw - raw.max()) / temp, dim=-1)
    masked = torch.full((logits.shape[0], logits.shape[1]), -1000000000.0, device=logits.device, dtype=logits.dtype)
    masked[:, feas] = feasible_logits
    log_p = torch.log_softmax(masked, dim=-1)[:, feas]
    return torch.sum(target * (torch.log(target.clamp_min(1e-08)) - log_p.squeeze(0)))

class VectorEIGEngine:
    """CUDA-batched posterior and expected one-step EIG calculations."""

    def __init__(self, ctx: ExperimentContext, device: torch.device):
        self.ctx = ctx
        self.device = device
        self.centres = torch.as_tensor(np.transpose(ctx.centres_support, (1, 0, 2)), dtype=torch.float32, device=device)
        self.log_p0 = torch.as_tensor(ctx.log_p0, dtype=torch.float32, device=device)
        self.sigma = float(ctx.sigma_y)
        self.sigma2 = self.sigma ** 2

    @staticmethod
    def entropy(log_w: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(log_w, dim=-1)
        return -(p * torch.log(p.clamp_min(1e-30))).sum(dim=-1)

    def update(self, log_w: torch.Tensor, action: int, observation: torch.Tensor) -> torch.Tensor:
        diff = self.centres[:, int(action), :] - observation
        return log_w - 0.5 * torch.sum(diff * diff, dim=-1) / self.sigma2

    @torch.no_grad()
    def action_scores(self, log_w: torch.Tensor, feasible: np.ndarray, *, n_fantasies: int, seed: int) -> np.ndarray:
        """Expected one-step entropy reduction for all feasible actions."""
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        p = torch.softmax(log_w, dim=-1)
        h0 = self.entropy(log_w)
        n_particles = len(p)
        sample_ids = torch.multinomial(p, int(n_fantasies), replacement=True, generator=generator)
        out = np.full(self.ctx.n_actions, -np.inf, dtype=np.float64)
        for start in range(0, len(feasible), 32):
            acts_np = feasible[start:start + 32]
            acts = torch.as_tensor(acts_np, dtype=torch.long, device=self.device)
            clean = self.centres[sample_ids[:, None], acts[None, :], :]
            clean = clean.permute(1, 0, 2)
            noise = torch.randn(clean.shape, generator=generator, device=self.device, dtype=clean.dtype) * self.sigma
            y = clean + noise
            centres = self.centres[:, acts, :].permute(1, 0, 2)
            distances = torch.cdist(y, centres, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
            quad = distances * distances
            ll = -0.5 * quad / self.sigma2
            post_h = self.entropy(log_w[None, None, :] + ll)
            gains = h0 - post_h.mean(dim=1)
            if not bool(torch.isfinite(gains).all()):
                raise RuntimeError(f'Non-finite one-step EIG encountered; refusing to train on invalid targets (actions={acts_np.tolist()}, seed={seed}).')
            out[acts_np] = gains.detach().cpu().numpy()
        return out

    @torch.no_grad()
    def two_step_scores(self, log_w: torch.Tensor, feasible: np.ndarray, *, n_fantasies: int, seed: int, has_future_step: bool) -> np.ndarray:
        """One-step EIG plus a posterior-conditioned continuation value.

        The score is defined for every feasible action, so the selected action
        is not restricted to any expert's top-1 proposal.
        """
        immediate = self.action_scores(log_w, feasible, n_fantasies=n_fantasies, seed=seed)
        if not has_future_step or len(feasible) <= 1:
            return immediate
        generator = torch.Generator(device=self.device).manual_seed(int(seed) + 37)
        p = torch.softmax(log_w, dim=-1)
        sample_ids = torch.multinomial(p, max(2, int(n_fantasies) // 2), replacement=True, generator=generator)
        out = immediate.copy()
        for action in feasible:
            clean = self.centres[sample_ids, int(action), :]
            noise = torch.randn(clean.shape, generator=generator, device=self.device, dtype=clean.dtype) * self.sigma
            continuation = []
            if str(getattr(self.ctx, 'observation_mode', '')).startswith('sir_'):
                next_feasible = feasible[feasible > int(action)]
            else:
                next_feasible = feasible[feasible != int(action)]
            if next_feasible.size == 0:
                continue
            for fantasy_id, observation in enumerate(clean + noise):
                post = self.update(log_w, int(action), observation)
                next_scores = self.action_scores(post, next_feasible, n_fantasies=max(2, int(n_fantasies) // 2), seed=int(seed) + 1009 * (int(action) + 1) + fantasy_id)
                best_next = float(np.max(next_scores[next_feasible]))
                if not np.isfinite(best_next):
                    raise RuntimeError(f'Non-finite two-step EIG continuation encountered (action={int(action)}, seed={seed}).')
                continuation.append(best_next)
            continuation_mean = float(np.mean(continuation))
            if not np.isfinite(continuation_mean):
                raise RuntimeError(f'Non-finite mean two-step EIG continuation encountered (action={int(action)}, seed={seed}).')
            out[int(action)] += continuation_mean
        if not np.all(np.isfinite(out[feasible])):
            raise RuntimeError(f'Non-finite two-step EIG scores encountered for feasible actions (seed={seed}).')
        return out

def _policy_tensors(ctx: ExperimentContext, actions: list[int], observations: list[np.ndarray], log_w: torch.Tensor, *, step: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """EIG policy utilities; historical tensor shapes are retained for saved checkpoints."""
    from src.policies.state import _tensors_from_state
    tensors = _tensors_from_state(ctx, actions=actions, observations=observations, log_w=log_w.detach().cpu().numpy(), step=step, device=device)
    action_idx, obs, mask, belief, steps, particles, weights, feasible = tensors
    expected_actions = _eig_feasible(ctx, actions, remaining_steps=max(int(ctx.horizon) - int(step), 0))
    expected_feasible = torch.zeros_like(feasible)
    expected_feasible[0, torch.as_tensor(expected_actions, dtype=torch.long, device=device)] = True
    if not torch.equal(feasible, expected_feasible):
        raise RuntimeError(f'EIG scoring and policy feasible-action masks disagree at step={step}, history={actions}.')
    belief = belief.clone()
    belief[..., 7:29] = 0.0
    particles = particles.clone()
    if particles.shape[-1] == 3:
        particles[..., 2] = 0.0
    return (action_idx, obs, mask, belief, steps, particles, weights, feasible)

def _observe(system: dict[str, Any], action: int, *, sigma: float, rollout_id: int, step: int, eval_seed: int | None=None) -> np.ndarray:
    """Additive Gaussian noise. Training keeps GLOBAL_SEED; eval uses eval_seed."""
    clean = np.asarray(system['obs_clean'][int(action)], dtype=np.float32)
    base = int(GLOBAL_SEED if eval_seed is None else eval_seed)
    rng = np.random.default_rng(base + 97451 * int(rollout_id) + 104729 * int(step))
    return clean + rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)

def _load_policy(ctx: ExperimentContext, name: str, device: torch.device) -> AdaptiveExperimentPolicy:
    path = model_dir(ctx.out_dir) / f'{name}.pth'
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = dict(payload.get('meta', {}))
    policy_cls = AdaptiveExperimentPolicy
    training = ctx.cfg.training_for(getattr(ctx, 'experiment_type', 'eig_based'))
    hidden = int(meta.get('policy_hidden') or training.get('policy_hidden', 128))
    config = PolicyConfig(max_steps=ctx.horizon, obs_dim=ctx.obs_dim, summary_dim=33, hidden=hidden, particle_dim=int(ctx.particle_features.shape[1]))
    policy = policy_cls(ctx.n_actions, config).to(device)
    sd = payload['state_dict']
    policy.load_state_dict(sd)
    policy.eval()
    return policy

def train_vector_eig_policy(ctx: ExperimentContext, *, method: str, smoke: bool, seed: int) -> dict[str, Any]:
    """EIG policy utilities; historical tensor shapes are retained for saved checkpoints."""
    if method not in {'dad_eig', 'rl_sboed_eig'}:
        raise ValueError(method)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = VectorEIGEngine(ctx, device)
    training = ctx.cfg.training_for(getattr(ctx, 'experiment_type', 'eig_based'))
    hidden = int(training.get('policy_hidden', 128))
    config = PolicyConfig(max_steps=ctx.horizon, obs_dim=ctx.obs_dim, summary_dim=33, hidden=hidden, particle_dim=int(ctx.particle_features.shape[1]))
    policy = AdaptiveExperimentPolicy(ctx.n_actions, config).to(device)
    epochs = 2 if smoke else int(training.get('eig_epochs', 20))
    steps_per_epoch = 16 if smoke else int(training.get('eig_steps_per_epoch', len(ctx.train_systems)))
    batch_size = 4 if smoke else int(training.get('batch_size', 16))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(training.get('learning_rate', 0.001)), weight_decay=0.0001)
    critic = None
    critic_optimizer = None
    ppo_epochs = 2 if smoke else int(training.get('eig_ppo_epochs', 4))
    ppo_clip = float(training.get('eig_ppo_clip', 0.2))
    entropy_coef = float(training.get('entropy_coef', 0.01))
    bc_temperature = float(training.get('eig_bc_temperature', 0.5))
    bc_lookahead = str(training.get('eig_bc_lookahead', 'two_step')).lower()
    bc_fantasies = int(training.get('eig_bc_fantasies', 16 if not smoke else 4))
    rl_use_ppo = bool(training.get('eig_rl_use_ppo', True))
    dad_use_ppo = bool(training.get('eig_dad_use_ppo', False))
    use_actor_critic = method == 'rl_sboed_eig' and rl_use_ppo or (method == 'dad_eig' and dad_use_ppo)
    if use_actor_critic:
        critic = StateValueCritic(ctx.n_actions, PolicyConfig(max_steps=ctx.horizon, obs_dim=ctx.obs_dim, summary_dim=33, hidden=hidden, particle_dim=int(ctx.particle_features.shape[1]))).to(device)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=float(training.get('eig_critic_lr', 0.001)), weight_decay=0.0001)
    min_unique_frac = float(training.get('eig_min_unique_sequence_fraction', 0.05))
    unique_eig_slack = float(training.get('eig_unique_floor_slack', 0.02))
    prefer_unique_floor = bool(training.get('eig_prefer_unique_sequence_floor', True))
    rng = np.random.default_rng(int(seed))
    baseline = np.zeros(ctx.horizon, dtype=np.float64)
    history = []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    bc_trajectories = 12 if smoke else int(training.get('eig_bc_trajectories', 128))
    bc_losses = []
    policy.train()
    print(f'[eig:{method}] BC warm-start trajectories={bc_trajectories} horizon={ctx.horizon} n_actions={ctx.n_actions} lookahead={bc_lookahead} temp={bc_temperature} actor_critic={use_actor_critic} obs_seed={int(seed)}', flush=True)
    for bc_id in range(bc_trajectories):
        if bc_id == 0 or (bc_id + 1) % max(1, bc_trajectories // 8) == 0 or bc_id + 1 == bc_trajectories:
            print(f'[eig:{method}] BC {bc_id + 1}/{bc_trajectories}', flush=True)
        system = ctx.train_systems[int(rng.integers(len(ctx.train_systems)))]
        actions: list[int] = []
        observations: list[np.ndarray] = []
        log_w = engine.log_p0.clone()
        trajectory_losses = []
        for step in range(ctx.horizon):
            feasible = _eig_feasible(ctx, actions)
            if bc_lookahead in {'two_step', '2step', 'two-step'}:
                scores = engine.two_step_scores(log_w, feasible, n_fantasies=bc_fantasies, seed=int(seed) + bc_id * 1009 + step, has_future_step=step < ctx.horizon - 1)
            else:
                scores = engine.action_scores(log_w, feasible, n_fantasies=bc_fantasies, seed=int(seed) + bc_id * 1009 + step)
            label = int(np.argmax(scores))
            tensors = _policy_tensors(ctx, actions, observations, log_w, step=step, device=device)
            logits = policy(*tensors)
            imitation = _soft_bc_loss(logits, scores, feasible, temperature=bc_temperature)
            trajectory_losses.append(imitation)
            y_np = _observe(system, label, sigma=ctx.sigma_y, rollout_id=50000 + bc_id, step=step, eval_seed=int(seed))
            log_w = engine.update(log_w, label, torch.as_tensor(y_np, device=device))
            actions.append(label)
            observations.append(y_np)
        optimizer.zero_grad(set_to_none=True)
        bc_loss = torch.stack(trajectory_losses).mean()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        bc_losses.append(float(bc_loss.detach().item()))
    n_validation = 4 if smoke else int(training.get('eig_validation_systems', 128))
    validation_systems = ctx.validation_systems[:n_validation]
    validation_fixed = _fixed_sequence(ctx, engine, n_fantasies=4 if smoke else 12, seed=int(seed))

    def validation_eig() -> tuple[float, int]:
        policy.eval()
        rollouts = [_rollout(ctx, engine, system, rollout_id=70000 + i, method=method, dad=policy, fixed_sequence=validation_fixed, n_fantasies=4 if smoke else 12, eval_seed=int(seed)) for i, system in enumerate(validation_systems)]
        policy.train()
        return (float(np.mean([row['terminal_eig'] for row in rollouts])), len({tuple(row['sequence']) for row in rollouts}))
    best_validation_eig, best_validation_unique = validation_eig()
    n_val_rollouts = max(len(validation_systems), 1)
    minimum_unique = max(1, int(math.ceil(min_unique_frac * n_val_rollouts)))
    best_meets_adaptivity = best_validation_unique >= minimum_unique
    best_admissible_eig = best_validation_eig if best_meets_adaptivity else float('-inf')
    best_stage = 'behavioral_cloning'
    best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    fallback_eig = best_validation_eig
    fallback_unique = best_validation_unique
    fallback_stage = best_stage
    fallback_state = best_state
    for epoch in range(epochs):
        epoch_gains = []
        epoch_losses = []
        for batch_start in range(0, steps_per_epoch, batch_size):
            losses = []
            ppo_states: list[tuple[torch.Tensor, ...]] = []
            ppo_actions: list[torch.Tensor] = []
            ppo_old_log_probs: list[torch.Tensor] = []
            ppo_returns: list[float] = []
            for sample_offset in range(min(batch_size, steps_per_epoch - batch_start)):
                rollout_id = epoch * steps_per_epoch + batch_start + sample_offset
                system = ctx.train_systems[int(rng.integers(len(ctx.train_systems)))]
                actions: list[int] = []
                observations: list[np.ndarray] = []
                log_probs = []
                entropies = []
                rewards = []
                trajectory_states: list[tuple[torch.Tensor, ...]] = []
                trajectory_actions: list[torch.Tensor] = []
                trajectory_old_lp: list[torch.Tensor] = []
                log_w = engine.log_p0.clone()
                entropy_before = float(engine.entropy(log_w).item())
                for step in range(ctx.horizon):
                    tensors = _policy_tensors(ctx, actions, observations, log_w, step=step, device=device)
                    if use_actor_critic:
                        with torch.no_grad():
                            dist = policy.distribution(*tensors)
                            action_t = dist.sample()
                            log_prob = dist.log_prob(action_t)
                            entropy = dist.entropy()
                        trajectory_states.append(tuple((t.detach() for t in tensors)))
                        trajectory_actions.append(action_t.detach())
                        trajectory_old_lp.append(log_prob.detach())
                    else:
                        dist = policy.distribution(*tensors)
                        action_t = dist.sample()
                        log_prob = dist.log_prob(action_t)
                        entropy = dist.entropy()
                    action = int(action_t.item())
                    y_np = _observe(system, action, sigma=ctx.sigma_y, rollout_id=rollout_id, step=step, eval_seed=int(seed))
                    y = torch.as_tensor(y_np, device=device)
                    log_w = engine.update(log_w, action, y)
                    entropy_after = float(engine.entropy(log_w).item())
                    rewards.append(entropy_before - entropy_after)
                    entropy_before = entropy_after
                    actions.append(action)
                    observations.append(y_np)
                    log_probs.append(log_prob.squeeze(0))
                    entropies.append(entropy.squeeze(0))
                rewards_a = np.asarray(rewards)
                returns = np.asarray([rewards_a.sum()] * ctx.horizon) if method == 'dad_eig' else np.cumsum(rewards_a[::-1])[::-1].copy()
                if use_actor_critic:
                    ppo_states.extend(trajectory_states)
                    ppo_actions.extend(trajectory_actions)
                    ppo_old_log_probs.extend(trajectory_old_lp)
                    ppo_returns.extend((float(v) for v in returns))
                else:
                    baseline = 0.9 * baseline + 0.1 * returns
                    advantage = torch.as_tensor(returns - baseline, dtype=torch.float32, device=device)
                    lp = torch.stack(log_probs)
                    ent = torch.stack(entropies)
                    losses.append(-torch.sum(lp * advantage) - entropy_coef * ent.sum())
                epoch_gains.append(float(rewards_a.sum()))
            if use_actor_critic:
                assert critic is not None and critic_optimizer is not None
                inputs = tuple((torch.cat([state[i] for state in ppo_states], dim=0) for i in range(len(ppo_states[0]))))
                action_t = torch.cat(ppo_actions).long()
                old_lp_t = torch.cat(ppo_old_log_probs).detach()
                return_t = torch.as_tensor(ppo_returns, dtype=torch.float32, device=device)
                with torch.no_grad():
                    old_values = critic(*inputs[:-1])
                    advantage_t = return_t - old_values
                    advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-08)
                last_loss = None
                for _ in range(ppo_epochs):
                    dist = policy.distribution(*inputs)
                    new_lp = dist.log_prob(action_t)
                    ratio = torch.exp(new_lp - old_lp_t)
                    clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = -torch.min(ratio * advantage_t, clipped * advantage_t).mean()
                    policy_loss = policy_loss - entropy_coef * dist.entropy().mean()
                    actor_loss = policy_loss
                    optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    value_loss = torch.nn.functional.huber_loss(critic(*inputs[:-1]), return_t)
                    critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                    last_loss = actor_loss.detach()
                assert last_loss is not None
                epoch_losses.append(float(last_loss.item()))
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
        epoch_validation_eig, epoch_validation_unique = validation_eig()
        epoch_meets_adaptivity = epoch_validation_unique >= minimum_unique
        history.append({'epoch': epoch + 1, 'mean_terminal_eig': float(np.mean(epoch_gains)), 'mean_loss': float(np.mean(epoch_losses)), 'validation_terminal_eig': epoch_validation_eig, 'validation_n_unique_sequences': epoch_validation_unique, 'validation_meets_adaptivity': int(epoch_meets_adaptivity)})
        prefer_diverse_fallback = False
        better_fallback = epoch_validation_unique > fallback_unique or (epoch_validation_unique == fallback_unique and epoch_validation_eig > fallback_eig) if prefer_diverse_fallback else epoch_validation_eig > fallback_eig
        if better_fallback:
            fallback_eig = epoch_validation_eig
            fallback_unique = epoch_validation_unique
            fallback_stage = f'epoch_{epoch + 1}'
            fallback_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        if prefer_unique_floor:
            diverse_ok = epoch_meets_adaptivity and (epoch_validation_eig >= best_admissible_eig - unique_eig_slack or not best_meets_adaptivity)
            if diverse_ok and (epoch_validation_eig > best_admissible_eig or (not best_meets_adaptivity and epoch_validation_eig >= best_validation_eig - unique_eig_slack)):
                best_admissible_eig = epoch_validation_eig
                best_validation_eig = epoch_validation_eig
                best_validation_unique = epoch_validation_unique
                best_meets_adaptivity = True
                best_stage = f'epoch_{epoch + 1}'
                best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        elif epoch_meets_adaptivity and epoch_validation_eig > best_admissible_eig:
            best_admissible_eig = epoch_validation_eig
            best_validation_eig = epoch_validation_eig
            best_validation_unique = epoch_validation_unique
            best_meets_adaptivity = True
            best_stage = f'epoch_{epoch + 1}'
            best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    if not best_meets_adaptivity:
        best_validation_eig = fallback_eig
        best_validation_unique = fallback_unique
        best_stage = fallback_stage
        best_state = fallback_state
    policy.load_state_dict(best_state)
    policy.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    path = model_dir(ctx.out_dir) / f'{method}.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': policy.state_dict(), 'critic_state_dict': critic.state_dict() if critic is not None else None, 'meta': {'method': method, 'objective': 'stepwise_entropy_reduction' if method == 'rl_sboed_eig' else 'terminal_eig', 'training_seed': int(seed), 'policy_hidden': hidden, 'particle_dim': int(ctx.particle_features.shape[1]), 'obs_dim': ctx.obs_dim, 'n_obs': ctx.n_obs, 'policy_input': 'eig_information_only_spatial_latent_v2', 'policy_input_retained': 'history,ESS,max_weight,M_summary,K_summary,machinewise_MK_particles,posterior_weights,feasible_actions', 'policy_input_masked': 'U_quantiles,u_ctrl,U_level_masses', 'experiment_dir': str(ctx.out_dir.resolve()), 'architecture': 'dense_policy', 'n_experts': int(getattr(policy, 'n_experts', 0)), 'top_k': int(getattr(policy, 'top_k', 0)), 'expert_hidden': int(getattr(policy, 'expert_hidden', 0)), 'optimizer': 'ppo_actor_critic' if use_actor_critic else 'reinforce', 'ppo_epochs': ppo_epochs if use_actor_critic else 0, 'ppo_clip': ppo_clip if use_actor_critic else 0.0, 'returns': 'stepwise_returns_to_go' if method in {'rl_sboed_eig'} else 'terminal_broadcast', 'eig_bc_lookahead': bc_lookahead, 'eig_bc_temperature': bc_temperature, 'eig_rl_use_ppo': bool(use_actor_critic and method == 'rl_sboed_eig'), 'eig_dad_use_ppo': bool(use_actor_critic and method == 'dad_eig')}, 'elapsed_seconds': elapsed, 'history': history, 'behavioral_cloning': {'trajectories': bc_trajectories, 'mean_loss': float(np.mean(bc_losses))}, 'model_selection': {'criterion': 'held_out_terminal_eig', 'best_stage': best_stage, 'best_validation_terminal_eig': best_validation_eig, 'best_validation_n_unique_sequences': best_validation_unique, 'minimum_unique_sequences': minimum_unique, 'best_meets_adaptivity': best_meets_adaptivity, 'n_validation_systems': len(validation_systems)}}, path)
    return {'method': method, 'checkpoint': str(path), 'device': str(device), 'elapsed_seconds': elapsed, 'history': history, 'behavioral_cloning_trajectories': bc_trajectories, 'behavioral_cloning_mean_loss': float(np.mean(bc_losses)), 'best_stage': best_stage, 'best_validation_terminal_eig': best_validation_eig, 'best_validation_n_unique_sequences': best_validation_unique, 'minimum_unique_sequences': minimum_unique, 'best_meets_adaptivity': best_meets_adaptivity}

def _prior_two_step_action(ctx: ExperimentContext, engine: VectorEIGEngine, *, n_fantasies: int, seed: int) -> int:
    """EIG policy utilities; historical tensor shapes are retained for saved checkpoints."""
    feasible = _eig_feasible(ctx, [])
    scores = engine.action_scores(engine.log_p0.clone(), feasible, n_fantasies=n_fantasies, seed=int(seed))
    return int(np.argmax(scores))

def _fixed_sequence(ctx: ExperimentContext, engine: VectorEIGEngine, *, n_fantasies: int, seed: int | None=None) -> list[int]:
    """Open-loop Fixed: greedy one-step prior EIG (chronological on SIR)."""
    base = int(GLOBAL_SEED if seed is None else seed)
    seq: list[int] = []
    log_w = engine.log_p0.clone()
    for step in range(int(ctx.horizon)):
        feasible = _eig_feasible(ctx, seq)
        if feasible.size == 0:
            break
        scores = engine.action_scores(log_w, feasible, n_fantasies=n_fantasies, seed=base + step)
        seq.append(int(np.argmax(scores)))
    return seq

def _frozen_fixed_sequence(ctx: ExperimentContext, engine: VectorEIGEngine, *, n_fantasies: int) -> tuple[list[int], float, int, Path]:
    """Load or create the seed-independent EIG Fixed baseline artifact.

    The calibration seed belongs to the design calibration procedure, not to an
    evaluation replicate.  Keeping the artifact under ``model/`` preserves the
    existing result layout and makes every evaluation seed use the same design.
    """
    evaluation = dict(ctx.cfg.raw.get('evaluation') or {})
    calibration_seed = int(evaluation.get('eig_fixed_calibration_seed', 104729))
    path = model_dir(ctx.out_dir) / f'fixed_eig_T{int(ctx.horizon)}.json'
    expected = {'horizon': int(ctx.horizon), 'n_actions': int(ctx.n_actions), 'n_obs': int(ctx.n_obs), 'noise_sigma': float(ctx.sigma_y), 'calibration_seed': calibration_seed, 'n_fantasies': int(n_fantasies)}
    if path.is_file():
        raw = json.loads(path.read_text(encoding='utf-8'))
        for key, value in expected.items():
            if raw.get(key) != value:
                raise RuntimeError(f'Stale EIG Fixed artifact {path}: {key}={raw.get(key)!r}, expected {value!r}. Remove only this model artifact and rerun evaluation to recalibrate Fixed.')
        sequence = [int(a) for a in raw.get('selected_action_ids', [])]
        if len(sequence) != int(ctx.horizon) or len(set(sequence)) != len(sequence):
            raise RuntimeError(f'Invalid frozen EIG Fixed sequence in {path}: {sequence}')
        return (sequence, float(raw.get('elapsed_seconds', 0.0)), calibration_seed, path)
    started = time.perf_counter()
    sequence = _fixed_sequence(ctx, engine, n_fantasies=n_fantasies, seed=calibration_seed)
    if engine.device.type == 'cuda':
        torch.cuda.synchronize(engine.device)
    elapsed = float(time.perf_counter() - started)
    payload = {**expected, 'selected_action_ids': sequence, 'search_mode': 'greedy_prior_eig_frozen', 'elapsed_seconds': elapsed, 'evaluation_seed_independent': True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return (sequence, elapsed, calibration_seed, path)

def _validate_eig_evaluation_data(ctx: ExperimentContext) -> dict[str, Any]:
    """EIG policy utilities; historical tensor shapes are retained for saved checkpoints."""
    support = np.asarray(ctx.centres_support)
    if support.ndim != 3 or support.shape[0] != int(ctx.n_actions):
        raise RuntimeError(f'Invalid EIG support shape {support.shape}; expected (actions, particles, obs).')
    if not np.isfinite(support).all():
        raise RuntimeError('EIG posterior support contains non-finite observations')
    if support.shape[1] < 2 or support.shape[2] < 1:
        raise RuntimeError(f'EIG support is too small for inference: {support.shape}')
    per_action_variance = np.var(support, axis=1).mean(axis=1)
    informative = int(np.count_nonzero(per_action_variance > 1e-14))
    if informative < int(ctx.horizon):
        raise RuntimeError(f'Only {informative}/{ctx.n_actions} actions vary across particles; cannot support a no-repeat horizon T={ctx.horizon}.')
    if len(ctx.test_systems) == 0:
        raise RuntimeError('EIG evaluation bank has no held-out test systems')
    return {'support_shape': [int(x) for x in support.shape], 'finite_support': True, 'informative_actions': informative, 'n_test_systems': int(len(ctx.test_systems))}

def _paired_eig_rows(all_rows: list[dict[str, Any]], *, eval_seed: int) -> list[dict[str, Any]]:
    """Paired per-system EIG differences for one evaluation seed."""
    by_method_theta: dict[str, dict[int, list[float]]] = {}
    for row in all_rows:
        method = str(row['method'])
        theta_id = int(row['theta_id'])
        by_method_theta.setdefault(method, {}).setdefault(theta_id, []).append(float(row['terminal_eig']))
    comparisons = (('step_dad', 'fixed_open_loop'), ('step_dad', 'myopic_delta_h'), ('step_dad', 'random'), ('step_dad', 'dad_eig'), ('dad_eig', 'fixed_open_loop'), ('dad_eig', 'myopic_delta_h'), ('dad_eig', 'random'), ('rl_sboed_eig', 'fixed_open_loop'), ('rl_sboed_eig', 'myopic_delta_h'), ('rl_sboed_eig', 'random'))
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        if left not in by_method_theta or right not in by_method_theta:
            continue
        theta_ids = sorted(set(by_method_theta[left]) & set(by_method_theta[right]))
        diffs = np.asarray([float(np.mean(by_method_theta[left][i])) - float(np.mean(by_method_theta[right][i])) for i in theta_ids], dtype=np.float64)
        if diffs.size == 0:
            continue
        sem = float(diffs.std(ddof=1) / math.sqrt(diffs.size)) if diffs.size > 1 else 0.0
        mean = float(diffs.mean())
        output.append({'comparison': f'{left} - {right}', 'left_method': left, 'right_method': right, 'n_paired_systems': int(diffs.size), 'mean_diff': mean, 'ci95_low': mean - 1.96 * sem, 'ci95_high': mean + 1.96 * sem, 'win_fraction': float(np.mean(diffs > 0.0)), 'eval_seed': int(eval_seed), 'pairing_note': 'paired by held-out theta; randomized baseline is averaged over its design replicates before differencing'})
    return output

@torch.no_grad()
def _rollout(ctx: ExperimentContext, engine: VectorEIGEngine, system: dict[str, Any], *, rollout_id: int, method: str, dad: AdaptiveExperimentPolicy | None, fixed_sequence: list[int], n_fantasies: int, eval_seed: int | None=None) -> dict[str, Any]:
    actions: list[int] = []
    observations: list[np.ndarray] = []
    log_w = engine.log_p0.clone()
    h0 = float(engine.entropy(log_w).item())
    step_eig = []
    trace = []
    rng_base = int(GLOBAL_SEED if eval_seed is None else eval_seed)
    for step in range(ctx.horizon):
        feasible = _eig_feasible(ctx, actions)
        myopic_scores = None
        if method == 'myopic_delta_h':
            myopic_scores = engine.action_scores(log_w, feasible, n_fantasies=n_fantasies, seed=rng_base + 1009 * rollout_id + step)
        if feasible.size == 0:
            raise RuntimeError(f'No chronologically feasible actions left at step {step} (history={actions}). SIR requires ξ1 < ξ2 < … < ξT.')
        if method == 'random':
            rng = np.random.default_rng(rng_base + rollout_id * 101 + step)
            action = int(rng.choice(feasible))
        elif method == 'fixed_open_loop':
            feasible_set = {int(a) for a in feasible.tolist()}
            action = next((int(a) for a in fixed_sequence if int(a) in feasible_set))
        elif method == 'myopic_delta_h':
            action = int(np.argmax(myopic_scores))
        else:
            assert dad is not None
            tensors = _policy_tensors(ctx, actions, observations, log_w, step=step, device=engine.device)
            logits = dad(*tensors).squeeze(0)
            action = int(torch.argmax(logits).item())
            if int(action) not in {int(a) for a in feasible.tolist()}:
                action = int(max(feasible.tolist(), key=lambda a: float(logits[int(a)])))
        y_np = _observe(system, action, sigma=ctx.sigma_y, rollout_id=rollout_id, step=step, eval_seed=eval_seed)
        h_before = float(engine.entropy(log_w).item())
        log_w = engine.update(log_w, action, torch.as_tensor(y_np, device=engine.device))
        h_after = float(engine.entropy(log_w).item())
        step_eig.append(h_before - h_after)
        actions.append(action)
        observations.append(y_np)
    return {'sequence': actions, 'terminal_eig': h0 - float(engine.entropy(log_w).item()), 'step_eig': step_eig, 'router_trace': trace}

def _step_dad_eig_config(ctx: ExperimentContext, smoke: bool) -> dict[str, Any]:
    raw = dict(ctx.cfg.training_for('eig_based') or {})
    refine = raw.get('eig_step_dad_refine_from_step')
    return {'updates': 2 if smoke else int(raw.get('eig_step_dad_refinement_steps', 64)), 'fantasies': 4 if smoke else int(raw.get('eig_step_dad_fantasy_rollouts', 16)), 'learning_rate': float(raw.get('eig_step_dad_learning_rate', 0.0001)), 'entropy': float(raw.get('eig_step_dad_entropy_coefficient', 0.001)), 'kl': float(raw.get('eig_step_dad_kl_coefficient', 1.0)), 'refine_at': max(1, int(ctx.horizon) // 2) if refine is None else int(refine)}

def _refine_step_dad_eig(ctx: ExperimentContext, engine: VectorEIGEngine, base: AdaptiveExperimentPolicy, *, actions: list[int], observations: list[np.ndarray], log_w: torch.Tensor, config: dict[str, Any], seed: int) -> tuple[AdaptiveExperimentPolicy, float]:
    """Original Step-DAD infer--refine update with discrete REINFORCE."""
    policy = copy.deepcopy(base).to(engine.device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config['learning_rate']))
    rng = np.random.default_rng(int(seed))
    started = time.perf_counter()
    posterior = torch.softmax(log_w, dim=-1).detach().cpu().numpy()
    initial_h = float(engine.entropy(log_w).detach())
    for _ in range(int(config['updates'])):
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        divergences: list[torch.Tensor] = []
        for _fantasy in range(int(config['fantasies'])):
            particle = int(rng.choice(len(posterior), p=posterior))
            fa = list(actions)
            fy = [np.asarray(y).copy() for y in observations]
            fw = log_w.detach().clone()
            lp: list[torch.Tensor] = []
            ent: list[torch.Tensor] = []
            for step in range(len(fa), int(ctx.horizon)):
                tensors = _policy_tensors(ctx, fa, fy, fw, step=step, device=engine.device)
                dist = policy.distribution(*tensors)
                with torch.no_grad():
                    base_dist = base.distribution(*tensors)
                divergences.append(torch.distributions.kl_divergence(base_dist, dist).mean())
                action_t = dist.sample()
                action = int(action_t.item())
                lp.append(dist.log_prob(action_t).reshape(()))
                ent.append(dist.entropy().reshape(()))
                clean = engine.centres[particle, action].detach().cpu().numpy()
                y = clean + float(ctx.sigma_y) * rng.normal(size=clean.shape)
                fa.append(action)
                fy.append(np.asarray(y, dtype=np.float32))
                fw = engine.update(fw, action, torch.as_tensor(y, dtype=torch.float32, device=engine.device))
            rewards.append(initial_h - float(engine.entropy(fw).detach()))
            log_probs.append(torch.stack(lp).sum())
            entropies.append(torch.stack(ent).sum())
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=engine.device)
        advantage = (reward_t - reward_t.mean()) / reward_t.std(unbiased=False).clamp_min(1e-06)
        loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        loss = loss - float(config['entropy']) * torch.stack(entropies).mean()
        loss = loss + float(config['kl']) * torch.stack(divergences).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    validation_rng = np.random.default_rng(int(seed) + 9999991)
    validation_cases = []
    for _ in range(max(8, int(config['fantasies']))):
        particle = int(validation_rng.choice(len(posterior), p=posterior))
        noises = [validation_rng.normal(size=engine.centres.shape[-1]) for _ in range(len(actions), int(ctx.horizon))]
        validation_cases.append((particle, noises))

    def validation_utility(candidate: AdaptiveExperimentPolicy) -> float:
        values = []
        candidate.eval()
        for particle, noises in validation_cases:
            fa = list(actions)
            fy = [np.asarray(y).copy() for y in observations]
            fw = log_w.detach().clone()
            for offset, step in enumerate(range(len(fa), int(ctx.horizon))):
                tensors = _policy_tensors(ctx, fa, fy, fw, step=step, device=engine.device)
                with torch.no_grad():
                    action = int(candidate(*tensors).argmax(dim=-1).item())
                clean = engine.centres[particle, action].detach().cpu().numpy()
                y = clean + float(ctx.sigma_y) * noises[offset]
                fa.append(action)
                fy.append(np.asarray(y, dtype=np.float32))
                fw = engine.update(fw, action, torch.as_tensor(y, dtype=torch.float32, device=engine.device))
            values.append(initial_h - float(engine.entropy(fw).detach()))
        return float(np.mean(values))
    if validation_utility(policy) <= validation_utility(base):
        return (base, float(time.perf_counter() - started))
    policy.eval()
    return (policy, float(time.perf_counter() - started))

def _rollout_step_dad_eig(ctx: ExperimentContext, engine: VectorEIGEngine, system: dict[str, Any], *, rollout_id: int, base: AdaptiveExperimentPolicy, config: dict[str, Any], eval_seed: int) -> dict[str, Any]:
    actions: list[int] = []
    observations: list[np.ndarray] = []
    log_w = engine.log_p0.clone()
    h0 = float(engine.entropy(log_w))
    step_eig: list[float] = []
    policy = base
    refined = False
    refine_at = min(max(int(config['refine_at']), 1), max(ctx.horizon - 1, 1))
    refinement_seconds = 0.0
    for step in range(ctx.horizon):
        if not refined and step == refine_at:
            policy, refinement_seconds = _refine_step_dad_eig(ctx, engine, base, actions=actions, observations=observations, log_w=log_w, config=config, seed=int(eval_seed) + 100003 * rollout_id)
            refined = True
        tensors = _policy_tensors(ctx, actions, observations, log_w, step=step, device=engine.device)
        with torch.no_grad():
            action = int(torch.argmax(policy(*tensors), dim=-1).item())
        y_np = _observe(system, action, sigma=ctx.sigma_y, rollout_id=rollout_id, step=step, eval_seed=eval_seed)
        before = float(engine.entropy(log_w))
        log_w = engine.update(log_w, action, torch.as_tensor(y_np, device=engine.device))
        step_eig.append(before - float(engine.entropy(log_w)))
        actions.append(action)
        observations.append(y_np)
    return {'sequence': actions, 'terminal_eig': h0 - float(engine.entropy(log_w)), 'step_eig': step_eig, 'router_trace': [], 'step_dad_refine_at': int(refine_at), 'step_dad_refinement_seconds': float(refinement_seconds), 'step_dad_refinement_updates': int(config['updates']), 'step_dad_fantasy_rollouts': int(config['updates'] * config['fantasies'])}
_VECTOR_CHECKPOINTS = {'dad_eig': 'dad_eig.pth', 'rl_sboed_eig': 'rl_sboed_eig.pth'}

def evaluate_vector_eig(ctx: ExperimentContext, *, smoke: bool, methods: tuple[str, ...] | None=None, eval_seed: int | None=None) -> dict[str, Any]:
    """Evaluate selected methods with common vector noise and write compact tables."""
    from src.layout import resolve_eval_seed
    if eval_seed is None:
        eval_seed = resolve_eval_seed(ctx.out_dir)
    eval_seed = int(eval_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = VectorEIGEngine(ctx, device)
    eig_data_audit = _validate_eig_evaluation_data(ctx)
    available_methods = METHODS
    selected_methods = available_methods if methods is None else tuple(methods)
    unknown = sorted(set(selected_methods) - set(available_methods))
    if unknown:
        raise ValueError(f'Unavailable vector-EIG methods: {unknown}')
    kept: list[str] = []
    for method in selected_methods:
        ckpt = _VECTOR_CHECKPOINTS.get(method)
        if ckpt is not None and (not (model_dir(ctx.out_dir) / ckpt).is_file()):
            print(f'[evaluate] skip {method}: missing {ckpt}')
            continue
        kept.append(method)
    selected_methods = tuple(kept)
    if not selected_methods:
        raise ValueError('No vector-EIG methods left to evaluate')
    dad = _load_policy(ctx, 'dad_eig', device) if 'dad_eig' in selected_methods or 'step_dad' in selected_methods else None
    rl = _load_policy(ctx, 'rl_sboed_eig', device) if 'rl_sboed_eig' in selected_methods else None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    fixed: tuple[int, ...] = ()
    fixed_preparation_seconds = 0.0
    fixed_calibration_seed: int | None = None
    fixed_artifact: str | None = None
    if 'fixed_open_loop' in selected_methods:
        fixed, fixed_preparation_seconds, fixed_calibration_seed, fixed_path = _frozen_fixed_sequence(ctx, engine, n_fantasies=4 if smoke else 16)
        fixed_artifact = str(fixed_path)
    training_seconds = {}
    for method_name, checkpoint_name in (('dad_eig', 'dad_eig'), ('rl_sboed_eig', 'rl_sboed_eig')):
        if method_name in selected_methods:
            payload = torch.load(model_dir(ctx.out_dir) / f'{checkpoint_name}.pth', map_location='cpu', weights_only=False)
            training_seconds[method_name] = float(payload.get('elapsed_seconds', 0.0))
    if 'step_dad' in selected_methods:
        dad_payload = torch.load(model_dir(ctx.out_dir) / 'dad_eig.pth', map_location='cpu', weights_only=False)
        training_seconds['step_dad'] = float(dad_payload.get('elapsed_seconds', 0.0))
    if 'fixed_open_loop' in selected_methods:
        training_seconds['fixed_open_loop'] = fixed_preparation_seconds
    evaluation = dict(ctx.cfg.raw.get('evaluation') or {})
    n = min(4 if smoke else int(evaluation.get('eig_test_systems', 128)), len(ctx.test_systems))
    summaries = []
    all_rows = []
    for method in selected_methods:
        policy = rl if method == 'rl_sboed_eig' else dad
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        random_replicates = int(evaluation.get('eig_random_replicates', 32))
        replicates = (8 if smoke else random_replicates) if method == 'random' else 1
        rows = []
        step_dad_config = _step_dad_eig_config(ctx, smoke)
        for i in range(n):
            for replicate in range(replicates):
                rollout_id = i * replicates + replicate
                if method == 'step_dad':
                    assert dad is not None
                    row = _rollout_step_dad_eig(ctx, engine, ctx.test_systems[i], rollout_id=rollout_id, base=dad, config=step_dad_config, eval_seed=eval_seed)
                else:
                    row = _rollout(ctx, engine, ctx.test_systems[i], rollout_id=rollout_id, method=method, dad=policy if method not in {'random', 'fixed_open_loop', 'myopic_delta_h'} else None, fixed_sequence=fixed, n_fantasies=4 if smoke else 16, eval_seed=eval_seed)
                row['theta_id'] = i
                row['design_replicate'] = replicate
                rows.append(row)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed_method = float(time.perf_counter() - started)
        per_theta = np.asarray([np.mean([r['terminal_eig'] for r in rows if int(r['theta_id']) == i]) for i in range(n)], dtype=np.float64)
        sem = float(per_theta.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        summaries.append({'method': method, 'n': n, 'n_design_replicates': len(rows), 'design_replicates_per_system': int(replicates), 'design_replication_note': 'expected performance averaged over randomized designs' if method == 'random' else 'one deterministic design per held-out system', 'terminal_eig_mean': float(per_theta.mean()), 'terminal_eig_std': float(per_theta.std(ddof=1)) if n > 1 else 0.0, 'ci95_low': float(per_theta.mean() - 1.96 * sem), 'ci95_high': float(per_theta.mean() + 1.96 * sem), 'seconds': elapsed_method, 'offline_training_or_calibration_seconds': training_seconds.get(method, 0.0), 'training_time_seconds': training_seconds.get(method, 0.0), 'online_seconds_per_rollout': float(elapsed_method / max(len(rows), 1)), 'n_unique_sequences': len({tuple(r['sequence']) for r in rows}), 'eval_seed': int(eval_seed), 'timing_scope': 'offline=method-specific preparation; online=CUDA-synchronized warm action selection + observation lookup + posterior update; shared physical-bank generation excluded'})
        for i, row in enumerate(rows):
            all_rows.append({'method': method, 'rollout_id': i, **row})
    eval_dir = ctx.out_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / 'terminal_eig_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paired_rows = _paired_eig_rows(all_rows, eval_seed=eval_seed)
    observation_name = 'sir_infected_count' if str(getattr(ctx, 'observation_mode', '')).startswith('sir_') else 'sampled_delta_f_vector'
    (eval_dir / 'vector_eig_results.json').write_text(json.dumps({'observation': observation_name, 'n_obs': ctx.n_obs, 'device': str(device), 'eval_seed': int(eval_seed), 'eig_data_audit': eig_data_audit, 'fixed_calibration_seed': fixed_calibration_seed, 'fixed_artifact': fixed_artifact, 'random_design_replicates_per_system': 8 if smoke else int(evaluation.get('eig_random_replicates', 32)), 'paired_summaries': paired_rows, 'summaries': summaries, 'rollouts': all_rows}, indent=2) + '\n', encoding='utf-8')
    return {'observation': observation_name, 'n_obs': ctx.n_obs, 'device': str(device), 'eval_seed': int(eval_seed), 'eig_data_audit': eig_data_audit, 'fixed_calibration_seed': fixed_calibration_seed, 'paired_summaries': paired_rows, 'summaries': summaries}
