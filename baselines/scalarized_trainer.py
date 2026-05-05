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
import csv
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml

# Allow running directly from the repo root or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy
from models.critic import VectorCritic
from training.rollout import collect_episode, episode_summary


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


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
    n_episodes_per_update: int = 5,
    log_interval_override: int | None = None,
) -> None:
    """
    Train a scalarized A2C baseline for one fixed λ preset.

    Parameters
    ----------
    lam_name             : one of "safety", "speed", "comfort", "uniform"
    cfg                  : parsed default.yaml dict; loaded from disk if None
    checkpoint_dir       : root directory; checkpoints go to <checkpoint_dir>/<lam_name>/
    device               : torch device string
    n_episodes_per_update: episodes to collect before each gradient update (mini-batch)
    log_interval_override: print/CSV interval in episodes; overrides config value
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
    log_interval  = log_interval_override if log_interval_override is not None else t_cfg["log_interval"]
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

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(ckpt_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "G_safety", "G_speed", "G_comfort", "T", "L_policy", "L_critic"])

    # ── Training loop ─────────────────────────────────────────────────────────
    # Losses from the current accumulation window; cleared after each update.
    accumulated_losses: list[tuple[torch.Tensor, torch.Tensor]] = []
    # Most-recent averaged losses, used for logging between update boundaries.
    avg_L_policy: torch.Tensor = torch.tensor(0.0, device=device)
    avg_L_critic: torch.Tensor = torch.tensor(0.0, device=device)

    t0 = time.monotonic()

    try:
        for ep in range(1, n_episodes + 1):
            returns, log_probs, values, entropies, _crashed = collect_episode(
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

            accumulated_losses.append((L_policy, L_critic))

            # Update once per n_episodes_per_update episodes (or at the very end).
            if ep % n_episodes_per_update == 0 or ep == n_episodes:
                avg_L_policy = torch.stack([l[0] for l in accumulated_losses]).mean()
                avg_L_critic = torch.stack([l[1] for l in accumulated_losses]).mean()

                policy_opt.zero_grad()
                avg_L_policy.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                policy_opt.step()

                critic_opt.zero_grad()
                avg_L_critic.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                critic_opt.step()

                accumulated_losses = []

            if ep % log_interval == 0:
                s = episode_summary(returns)
                elapsed = time.monotonic() - t0
                eta = _fmt_eta(elapsed / ep * (n_episodes - ep))
                print(
                    f"[{lam_name}] ep {ep:5d}/{n_episodes}"
                    f"  G_safety={s['G_safety']:.3f}"
                    f"  G_speed={s['G_speed']:.3f}"
                    f"  G_comfort={s['G_comfort']:.3f}"
                    f"  T={s['length']:3d}"
                    f"  L_pol={avg_L_policy.item():.5f}"
                    f"  L_crit={avg_L_critic.item():.5f}"
                    f"  ETA {eta}",
                    flush=True,
                )
                csv_writer.writerow([
                    ep,
                    f"{s['G_safety']:.5f}",
                    f"{s['G_speed']:.5f}",
                    f"{s['G_comfort']:.5f}",
                    s["length"],
                    f"{avg_L_policy.item():.6f}",
                    f"{avg_L_critic.item():.6f}",
                ])
                csv_file.flush()

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
                print(f"  → saved {ckpt_path}", flush=True)
    finally:
        csv_file.close()
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
    parser.add_argument(
        "--n_episodes_per_update",
        type=int,
        default=5,
        help="Episodes to collect before each gradient update (default: 5)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
        help="Print/CSV interval in episodes; overrides config value (default: use config)",
    )
    args = parser.parse_args()

    presets_to_run = list(PRESETS) if args.all else [args.lam]
    for preset in presets_to_run:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Training baseline: {preset}  λ = {PRESETS[preset]}", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        train(
            lam_name=preset,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            n_episodes_override=args.episodes,
            n_episodes_per_update=args.n_episodes_per_update,
            log_interval_override=args.log_interval,
        )


if __name__ == "__main__":
    main()
