"""
Episode collection for preference-conditioned policies.

collect_episode() runs one full episode with ConditionedPolicy and VectorCritic,
returning four tensors needed for A2C training:

  returns   : Tensor (T, 3)  — discounted per-objective costs Gᵢ(t) = Σ γ^(t'−t) rᵢ(t')
  log_probs : Tensor (T,)    — log π(aₜ | sₜ, λ), kept in the computation graph
  values    : Tensor (T, 3)  — Vᵢ(sₜ, λ) from the critic, kept in the computation graph
  entropies : Tensor (T,)    — H(π(·|sₜ, λ)), kept in the computation graph

Bootstrap correction: when an episode ends by time-limit truncation (not crash),
the terminal state still has non-zero expected future cost. We bootstrap G from
the critic at the final observation rather than using zeros, which would
systematically underestimate returns for all non-terminal episodes.
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    entropies : Tensor (T,)    — H(π(·|sₜ, λ)), differentiable w.r.t. policy params
    """
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)

    obs, _ = env.reset()
    log_prob_list:  list[torch.Tensor] = []
    value_list:     list[torch.Tensor] = []
    entropy_list:   list[torch.Tensor] = []
    step_costs:     list[np.ndarray]   = []

    terminated = False
    truncated  = False
    done = False
    while not done:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)

        # Single forward pass yields action, log_prob, and entropy together.
        action, log_prob, entropy = policy.act(obs_t, lam_t, return_entropy=True)
        value = critic(obs_t, lam_t)                         # (3,)

        obs, cost_vec, terminated, truncated, _ = env.step(action)

        log_prob_list.append(log_prob)
        value_list.append(value)
        entropy_list.append(entropy)
        step_costs.append(cost_vec)
        done = terminated or truncated

    # --- Discounted returns per objective ---
    # When truncated (time limit), bootstrap terminal value from the critic so
    # we do not underestimate future cost. When terminated (crash), future cost = 0.
    T = len(step_costs)
    costs_arr  = np.stack(step_costs)                        # (T, 3)
    returns_np = np.zeros_like(costs_arr)

    if truncated:
        final_obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
        with torch.no_grad():
            G = critic(final_obs_t, lam_t).cpu().numpy()    # (3,) bootstrap
    else:
        G = np.zeros(3)                                      # crash → no future cost

    for t in reversed(range(T)):
        G = costs_arr[t] + gamma * G
        returns_np[t] = G

    returns   = torch.tensor(returns_np, dtype=torch.float32, device=device)  # (T,3), no grad
    log_probs = torch.stack(log_prob_list)                                     # (T,),  with grad
    values    = torch.stack(value_list)                                        # (T,3), with grad
    entropies = torch.stack(entropy_list)                                      # (T,),  with grad

    return returns, log_probs, values, entropies


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
