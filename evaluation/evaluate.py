"""
Evaluate a trained ConditionedPolicy across the preference simplex.

The whole point of PSL is that ONE network can be queried at any λ and produce
a Pareto-optimal policy for that preference. These helpers make it easy to:

  - score a single λ (mean per-objective costs over N episodes)
  - sweep a grid of λ values to map the learned Pareto surface

No retraining is involved — the policy is frozen and only the conditioning
vector λ changes.
"""

from __future__ import annotations

import numpy as np
import torch

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy


def _episode_costs(env: MOHighwayEnv, policy: ConditionedPolicy,
                   lam: np.ndarray, device: str = "cpu",
                   greedy: bool = False) -> tuple[np.ndarray, int]:
    """Run one episode under π(·, λ) and return mean per-step costs (3,) and length."""
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    obs, _ = env.reset()
    step_costs: list[np.ndarray] = []
    done = False
    while not done:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
        with torch.no_grad():
            if greedy:
                action = int(policy(obs_t, lam_t).argmax().item())
            else:
                action, _ = policy.act(obs_t, lam_t)
        obs, cost_vec, terminated, truncated, _ = env.step(action)
        step_costs.append(cost_vec)
        done = terminated or truncated
    arr = np.stack(step_costs)               # (T, 3)
    return arr.mean(axis=0), len(arr)


def evaluate_policy_at_lambda(
    policy: ConditionedPolicy,
    lam: np.ndarray,
    n_episodes: int = 5,
    env: MOHighwayEnv | None = None,
    device: str = "cpu",
    greedy: bool = False,
    seed: int | None = None,
) -> dict:
    """
    Evaluate a frozen policy at a single preference vector.

    Returns a dict with mean and std of per-objective costs (averaged over the
    episode timesteps for each episode, then averaged over episodes).
    """
    own_env = env is None
    env = env or MOHighwayEnv()
    if seed is not None:
        env.reset(seed=seed)

    policy.eval()
    per_ep_costs = []
    lengths = []
    for _ in range(n_episodes):
        c, T = _episode_costs(env, policy, lam, device=device, greedy=greedy)
        per_ep_costs.append(c)
        lengths.append(T)
    policy.train()

    if own_env:
        env.close()

    arr = np.stack(per_ep_costs)            # (n_episodes, 3)
    return {
        "lam":      lam,
        "mean":     arr.mean(axis=0),
        "std":      arr.std(axis=0),
        "all":      arr,
        "lengths":  np.array(lengths),
    }


def _simplex_grid(n_per_side: int) -> np.ndarray:
    """
    Triangular grid of preferences on the 3-simplex.
    Returns shape (n_points, 3) with sum=1 and components ≥ 0.
    """
    pts = []
    for i in range(n_per_side + 1):
        for j in range(n_per_side + 1 - i):
            k = n_per_side - i - j
            pts.append([i, j, k])
    pts = np.array(pts, dtype=np.float32) / float(n_per_side)
    return pts


def evaluate_policy_grid(
    policy: ConditionedPolicy,
    n_per_side: int = 6,
    n_episodes: int = 3,
    env: MOHighwayEnv | None = None,
    device: str = "cpu",
    greedy: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the policy at a triangular grid of λ on the 3-simplex.

    n_per_side=6 → 28 grid points (T(6+1) = 28). Each point runs n_episodes.

    Returns
    -------
    dict with keys:
        lambdas : (M, 3)  — preference vectors
        costs   : (M, 3)  — mean per-step costs at each λ
        stds    : (M, 3)
    """
    own_env = env is None
    env = env or MOHighwayEnv()

    lambdas = _simplex_grid(n_per_side)
    costs = np.zeros_like(lambdas)
    stds  = np.zeros_like(lambdas)

    for idx, lam in enumerate(lambdas):
        out = evaluate_policy_at_lambda(
            policy, lam,
            n_episodes=n_episodes,
            env=env, device=device, greedy=greedy,
        )
        costs[idx] = out["mean"]
        stds[idx]  = out["std"]
        if verbose:
            print(f"[{idx + 1:3d}/{len(lambdas)}] "
                  f"λ=({lam[0]:.2f},{lam[1]:.2f},{lam[2]:.2f}) → "
                  f"safety={costs[idx,0]:.3f}  speed={costs[idx,1]:.3f}  "
                  f"comfort={costs[idx,2]:.3f}")

    if own_env:
        env.close()

    return {"lambdas": lambdas, "costs": costs, "stds": stds}
