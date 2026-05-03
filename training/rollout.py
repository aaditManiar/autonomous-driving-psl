"""
Episode collection for preference-conditioned policies.

collect_episode() runs one full episode with ConditionedPolicy and VectorCritic,
returning the three tensors needed for A2C training:

  returns   : Tensor (T, 3)  — discounted per-objective costs Gᵢ(t) = Σ γ^(t'−t) rᵢ(t')
  log_probs : Tensor (T,)    — log π(aₜ | sₜ, λ), kept in the computation graph
  values    : Tensor (T, 3)  — Vᵢ(sₜ, λ) from the critic, kept in the computation graph

The caller computes advantages as A = returns − values and constructs the A2C
policy and critic losses from there.
"""

import numpy as np
import torch

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy
from models.critic import VectorCritic


def collect_episode(
    env: MOHighwayEnv,
    policy: ConditionedPolicy,
    critic: VectorCritic,
    lam: np.ndarray,
    gamma: float = 0.99,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one episode and collect the data needed for A2C training.

    Parameters
    ----------
    env    : MOHighwayEnv  — reset() is called at the start
    policy : ConditionedPolicy
    critic : VectorCritic
    lam    : np.ndarray (3,)  — preference vector, sums to 1
    gamma  : float  — discount factor
    device : str

    Returns
    -------
    returns   : Tensor (T, 3)  — discounted cumulative cost per objective, no grad
    log_probs : Tensor (T,)    — log π(aₜ | sₜ, λ), differentiable w.r.t. policy params
    values    : Tensor (T, 3)  — Vᵢ(sₜ, λ), differentiable w.r.t. critic params
    """
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)

    obs, _ = env.reset()
    log_prob_list: list[torch.Tensor] = []
    value_list: list[torch.Tensor] = []
    step_costs: list[np.ndarray] = []

    done = False
    while not done:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)

        action, log_prob = policy.act(obs_t, lam_t)
        value = critic(obs_t, lam_t)             # (3,)

        obs, cost_vec, terminated, truncated, _ = env.step(action)

        log_prob_list.append(log_prob)
        value_list.append(value)
        step_costs.append(cost_vec)
        done = terminated or truncated

    # --- Discounted returns per objective (backward pass over episode) ---
    T = len(step_costs)
    costs_arr = np.stack(step_costs)             # (T, 3)
    returns_np = np.zeros_like(costs_arr)

    G = np.zeros(3)
    for t in reversed(range(T)):
        G = costs_arr[t] + gamma * G
        returns_np[t] = G

    returns = torch.tensor(returns_np, dtype=torch.float32, device=device)  # (T, 3), no grad
    log_probs = torch.stack(log_prob_list)                                   # (T,), with grad
    values = torch.stack(value_list)                                         # (T, 3), with grad

    return returns, log_probs, values


def episode_summary(returns: torch.Tensor) -> dict:
    """
    Summarise per-objective cumulative costs for logging.
    returns : Tensor (T, 3)
    """
    G0 = returns[0]   # total discounted cost from t=0, shape (3,)
    return {
        "G_safety":  float(G0[0]),
        "G_speed":   float(G0[1]),
        "G_comfort": float(G0[2]),
        "G_total":   float(G0.sum()),
        "length":    len(returns),
    }
