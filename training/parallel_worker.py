"""
Parallel episode collection worker for PSL training.

Each worker receives the current policy/critic weights and one preference vector,
runs N episodes, and returns the per-objective Jacobian (policy) and gradient
(critic) as numpy arrays for aggregation in the main process.

Must be defined in an importable module (not inline in a notebook cell) because
macOS uses the 'spawn' multiprocessing start method, which requires the worker
function to be importable from scratch in each child process.
"""

import os
import sys

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _flat_grad(params):
    return torch.cat([
        (p.grad.detach().flatten() if p.grad is not None
         else torch.zeros_like(p).flatten())
        for p in params
    ])


# --------------------------------------------------------------------------- #
# Worker entry point
# --------------------------------------------------------------------------- #

def collect_pref_worker(args):
    """
    Run N episodes for one preference vector and compute gradients.

    All arguments are packed into a single tuple so this function works with
    ProcessPoolExecutor.map without needing starmap.

    Parameters
    ----------
    args : tuple of
        policy_sd_np  : dict[str, np.ndarray]  — policy state dict as numpy arrays
        critic_sd_np  : dict[str, np.ndarray]  — critic state dict as numpy arrays
        lam_np        : np.ndarray (3,)
        n_episodes    : int
        gamma         : float
        entropy_coef  : float
        policy_cfg    : dict — keys: obs_dim, lam_dim, hidden_dim, n_actions
        env_cfg       : dict — env config overrides
        seed          : int | None

    Returns
    -------
    J_k             : np.ndarray (3, n_var)   — mean policy Jacobian across N episodes
    v_k             : np.ndarray (3,)         — mean per-step objective costs (for EPO)
    critic_grad     : np.ndarray (n_cvar,)    — mean critic gradient across N episodes
    policy_loss_vec : np.ndarray (3,)         — mean per-objective policy loss (incl. entropy bonus)
    critic_loss     : float                   — mean MSE of (returns − values) across N episodes
    """
    (policy_sd_np, critic_sd_np, lam_np, n_episodes,
     gamma, entropy_coef, policy_cfg, env_cfg, seed) = args

    # Workers spawn fresh on macOS — add repo root so local imports work
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from models.policy import ConditionedPolicy
    from models.critic import VectorCritic
    from envs import MOHighwayEnv
    from training.rollout import collect_episode

    if seed is not None:
        np.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed) % (2 ** 32))

    # Build local models from state dicts
    policy = ConditionedPolicy(**policy_cfg)
    policy.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in policy_sd_np.items()})

    critic = VectorCritic(
        obs_dim=policy_cfg["obs_dim"],
        lam_dim=policy_cfg["lam_dim"],
        hidden_dim=policy_cfg["hidden_dim"],
    )
    critic.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in critic_sd_np.items()})

    env = MOHighwayEnv(config=env_cfg)
    n_var = sum(p.numel() for p in policy.parameters())

    jacobian_rows:    list[np.ndarray] = []
    obj_values:       list[np.ndarray] = []
    critic_grads:     list[np.ndarray] = []
    policy_loss_vecs: list[np.ndarray] = []
    critic_losses:    list[float]      = []

    for _ in range(n_episodes):
        returns, log_probs, values, entropies = collect_episode(
            env, policy, critic, lam_np, gamma=gamma
        )
        advantages = (returns - values).detach()       # (T, 3) — detached from critic graph

        T = log_probs.shape[0]
        weighted = advantages * log_probs.unsqueeze(1) # (T, 3)
        per_obj = weighted.mean(dim=0) - entropy_coef * entropies.mean()  # (3,)
        policy_loss_vecs.append(per_obj.detach().cpu().numpy().copy())

        # Per-step costs for EPO calibration (must be in [0, 1])
        if T > 1:
            rn = returns.detach().cpu().numpy()
            mean_step_cost = (rn[:-1] - gamma * rn[1:]).mean(axis=0)
        else:
            mean_step_cost = returns[0].detach().cpu().numpy()
        obj_values.append(mean_step_cost)

        # Critic gradient — graph is independent of policy (advantages are detached)
        critic.zero_grad()
        critic_loss = ((returns - values) ** 2).mean()
        critic_loss.backward()
        critic_grads.append(_flat_grad(critic.parameters()).numpy().copy())
        critic_losses.append(float(critic_loss.detach().cpu().item()))

        # Policy Jacobian — 3 separate backward passes, one per objective
        ep_jac = np.zeros((3, n_var), dtype=np.float32)
        for i in range(3):
            policy.zero_grad(set_to_none=False)
            per_obj[i].backward(retain_graph=(i < 2))
            ep_jac[i] = _flat_grad(policy.parameters()).numpy().copy()
        jacobian_rows.append(ep_jac)

    env.close()

    return (
        np.mean(jacobian_rows,    axis=0),   # (3, n_var)
        np.mean(obj_values,       axis=0),   # (3,)
        np.mean(critic_grads,     axis=0),   # (n_cvar,)
        np.mean(policy_loss_vecs, axis=0),   # (3,)
        float(np.mean(critic_losses)),       # scalar
    )