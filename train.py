"""
Single-λ REINFORCE training entry point.

Purpose: verify that the full RL loop works (env → rollout → gradient → update)
before adding multi-objective PSL complexity in Phase 3.

The reward signal is the λ-weighted negative of the cost vector:
    r_scalar(t) = -λ · [f_safety(t), f_speed(t), f_comfort(t)]

Because the env returns COSTS (lower = better) and RL maximises REWARDS, we
negate. Baseline subtraction keeps gradients zero-centred despite the costs
living in [0,1] (which would otherwise make all returns negative and push the
policy away from every action equally).

Usage:
    .venv/bin/python3 train.py
    .venv/bin/python3 train.py --lam 1.0 0.0 0.0   # safety-only
    .venv/bin/python3 train.py --episodes 1000
"""

import argparse
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from envs import MOHighwayEnv
from models import ConditionedPolicy
from models.critic import VectorCritic
from training.rollout import collect_episode, episode_summary


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", nargs=3, type=float, default=None,
                   help="Preference vector (3 floats, will be normalised). "
                        "Default: uniform (1/3, 1/3, 1/3).")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override configs/default.yaml train.n_episodes.")
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def reinforce_loss(
    log_probs: list[torch.Tensor],
    returns: np.ndarray,
    lam: np.ndarray,
    baseline: float,
) -> torch.Tensor:
    """
    Compute the REINFORCE policy-gradient loss for one episode.

    G_scalar(t) = λ · (-returns(t))   — scalarised reward return
    L = -mean_t [ (G_scalar(t) - baseline) * log π(aₜ | sₜ, λ) ]

    Negating costs → rewards ensures that REINFORCE pushes the policy
    towards actions that *reduce* the objectives.
    """
    # Scalarised reward returns: (T,)
    reward_returns = (-returns) @ lam           # negate costs → rewards

    # Advantage = return - baseline (zero-centres the signal)
    advantages = reward_returns - baseline      # (T,)

    # Stack log-probs into a single tensor for vectorised multiplication
    log_prob_tensor = torch.stack(log_probs)    # (T,)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)

    # REINFORCE loss (negative because we do gradient *descent* on loss
    # but want gradient *ascent* on expected return)
    loss = -(adv_tensor * log_prob_tensor).mean()
    return loss


def train(config: dict, lam: np.ndarray, n_episodes: int):
    device = "cpu"
    pc = config["policy"]
    tc = config["train"]

    policy = ConditionedPolicy(
        obs_dim=pc["obs_dim"],
        lam_dim=pc["lam_dim"],
        hidden_dim=pc["hidden_dim"],
        n_actions=pc["n_actions"],
    ).to(device)

    optimiser = torch.optim.Adam(policy.parameters(), lr=tc["learning_rate"])
    env = MOHighwayEnv()

    # Exponential moving average baseline — tracks expected scalarised return.
    # Starts at 0; warms up over the first few episodes.
    baseline: float = 0.0
    alpha = tc["baseline_momentum"]   # EMA update coefficient

    print(f"Training with λ = {lam}  ({n_episodes} episodes)\n")

    for ep in range(1, n_episodes + 1):
        # --- Collect episode ---
        log_probs, returns = collect_episode(
            env, policy, lam,
            gamma=tc["gamma"],
            device=device,
        )

        # Scalar return for this episode (t=0 gives full discounted sum)
        ep_scalar_return = float((-returns[0]) @ lam)

        # --- Update baseline (EMA of episode returns) ---
        baseline = (1 - alpha) * baseline + alpha * ep_scalar_return

        # --- Compute loss and update policy ---
        loss = reinforce_loss(log_probs, returns, lam, baseline)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimiser.step()

        # --- Logging ---
        if ep % tc["log_interval"] == 0:
            summary = episode_summary(returns)
            print(
                f"ep {ep:4d} | "
                f"loss={loss.item():+.4f} | "
                f"return={ep_scalar_return:+.3f} | "
                f"baseline={baseline:+.3f} | "
                f"safety={summary['G_safety']:.3f}  "
                f"speed={summary['G_speed']:.3f}  "
                f"comfort={summary['G_comfort']:.3f} | "
                f"len={summary['length']}"
            )

        if ep % tc["save_interval"] == 0:
            path = f"checkpoints/policy_ep{ep}.pt"
            import os; os.makedirs("checkpoints", exist_ok=True)
            torch.save(policy.state_dict(), path)
            print(f"  → saved {path}")

    env.close()
    return policy


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.lam is not None:
        lam = np.array(args.lam, dtype=np.float64)
        lam /= lam.sum()    # normalise so it's a valid probability vector
    else:
        lam = np.array([1/3, 1/3, 1/3])

    n_episodes = args.episodes or config["train"]["n_episodes"]

    train(config, lam, n_episodes)


if __name__ == "__main__":
    main()
