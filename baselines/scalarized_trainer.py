"""
Scalarized A2C baseline trainer.

Trains a single ConditionedPolicy + VectorCritic for a FIXED preference vector λ,
using multi-objective A2C with a scalarized advantage:

    a_scalar(t) = λ · (G(t) − V(s_t, λ))

Policy loss (cost minimisation, no negation):
    L_policy = mean_t [ a_scalar(t) · log π(aₜ | sₜ, λ) ]

Four λ presets are supported (see CLAUDE.md Phase 4):
    safety  : λ = (1, 0, 0)
    speed   : λ = (0, 1, 0)
    comfort : λ = (0, 0, 1)
    uniform : λ = (1/3, 1/3, 1/3)

Each preset writes checkpoints to checkpoints/<preset>/ep<NNNNN>.pt every
`save_interval` episodes so PSL vs. baseline comparisons use the same episode
budget.

Usage (CLI):
    python baselines/scalarized_trainer.py --lam uniform
    python baselines/scalarized_trainer.py --all
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

# Allow running directly from the repo root or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy
from models.critic import VectorCritic
from training.rollout import collect_episode, episode_summary


PRESETS: dict[str, np.ndarray] = {
    "safety":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "speed":   np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "comfort": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "uniform": np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
}


def _load_cfg(cfg_path: str | None = None) -> dict:
    if cfg_path is None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "default.yaml",
        )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def train(
    lam_name: str = "uniform",
    cfg: dict | None = None,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    n_episodes_override: int | None = None,
) -> None:
    """
    Train a scalarized A2C baseline for one fixed λ preset.

    Parameters
    ----------
    lam_name       : one of "safety", "speed", "comfort", "uniform"
    cfg            : parsed default.yaml dict; loaded from disk if None
    checkpoint_dir : root directory; checkpoints go to <checkpoint_dir>/<lam_name>/
    device         : torch device string
    """
    if lam_name not in PRESETS:
        raise ValueError(f"Unknown preset '{lam_name}'. Choose from: {list(PRESETS)}")

    lam = PRESETS[lam_name]
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)

    if cfg is None:
        cfg = _load_cfg()

    t_cfg = cfg["train"]
    p_cfg = cfg["policy"]

    n_episodes    = n_episodes_override if n_episodes_override is not None else t_cfg["n_episodes"]
    gamma         = t_cfg["gamma"]
    lr            = t_cfg["learning_rate"]
    critic_lr     = t_cfg["critic_lr"]
    entropy_coef  = t_cfg.get("entropy_coef", 0.01)
    log_interval  = t_cfg["log_interval"]
    save_interval = t_cfg["save_interval"]

    # ── Models ────────────────────────────────────────────────────────────────
    policy = ConditionedPolicy(
        obs_dim=p_cfg["obs_dim"],
        lam_dim=p_cfg["lam_dim"],
        hidden_dim=p_cfg["hidden_dim"],
        n_actions=p_cfg["n_actions"],
    ).to(device)

    critic = VectorCritic(
        obs_dim=p_cfg["obs_dim"],
        lam_dim=p_cfg["lam_dim"],
        hidden_dim=p_cfg["hidden_dim"],
    ).to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = os.path.join(checkpoint_dir, lam_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Environment ───────────────────────────────────────────────────────────
    env = MOHighwayEnv(config=cfg.get("env", {}))

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, n_episodes + 1):
        returns, log_probs, values, entropies = collect_episode(
            env, policy, critic, lam, gamma, device
        )

        # Advantages: cost-based.  Positive A → step was worse than expected.
        # Detach values so the policy loss does not back-prop into the critic.
        advantages = returns - values.detach()           # (T, 3)
        scalar_adv = (advantages * lam_t).sum(dim=-1)   # (T,)

        # Policy loss: cost minimisation + entropy bonus to prevent collapse.
        L_policy = (scalar_adv * log_probs).mean() - entropy_coef * entropies.mean()

        # Critic loss: MSE across all objectives and all timesteps.
        L_critic = ((returns - values) ** 2).mean()

        policy_opt.zero_grad()
        L_policy.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy_opt.step()

        critic_opt.zero_grad()
        L_critic.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        critic_opt.step()

        if ep % log_interval == 0:
            s = episode_summary(returns)
            print(
                f"[{lam_name}] ep {ep:4d}/{n_episodes}"
                f"  G_safety={s['G_safety']:.3f}"
                f"  G_speed={s['G_speed']:.3f}"
                f"  G_comfort={s['G_comfort']:.3f}"
                f"  L_pol={L_policy.item():.5f}"
                f"  L_crit={L_critic.item():.5f}"
                f"  T={s['length']}"
            )

        if ep % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ep{ep:05d}.pt")
            torch.save(
                {
                    "episode": ep,
                    "lam_name": lam_name,
                    "lam": lam,
                    "policy_state_dict": policy.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "policy_opt_state_dict": policy_opt.state_dict(),
                    "critic_opt_state_dict": critic_opt.state_dict(),
                },
                ckpt_path,
            )
            print(f"  → saved {ckpt_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalarized A2C baseline — trains one agent per fixed λ preset"
    )
    parser.add_argument(
        "--lam",
        choices=list(PRESETS),
        default="uniform",
        help="Preset preference vector (default: uniform)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all four presets sequentially",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Root directory for checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override n_episodes from config",
    )
    args = parser.parse_args()

    presets_to_run = list(PRESETS) if args.all else [args.lam]
    for preset in presets_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Training baseline: {preset}  λ = {PRESETS[preset]}")
        print(f"{'=' * 60}\n")
        train(
            lam_name=preset,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            n_episodes_override=args.episodes,
        )


if __name__ == "__main__":
    main()
