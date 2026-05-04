"""
Pareto Set Learning (PSL) trainer — LibMOON EPO bridge to multi-objective A2C.

A *single* preference-conditioned policy π(·, λ) is trained to be Pareto-optimal
across the whole preference simplex Λ. At inference time, varying λ produces
different driving styles WITHOUT retraining.

Algorithm (per update step):

  1. Sample K preference vectors λ_1, …, λ_K ~ Dirichlet(α=1) on the 2-simplex.
  2. For each λ_k:
       a. Run `n_episodes_per_pref` episodes with π(·, λ_k), collect:
              returns G_i(t) ∈ ℝ^(T,3),  log-probs ∈ ℝ^T,  values V_i(s_t,λ_k) ∈ ℝ^(T,3)
       b. Build per-objective policy losses (cost form, no negation since we minimise):
              L_i = mean_t [ A_i(t) · log π(a_t | s_t, λ_k) ]
              where A_i(t) = G_i(t) − V_i(s_t, λ_k)   (detached)
       c. Compute Jacobian J_k ∈ ℝ^(3,n_var) of L_i w.r.t. policy params.
       d. Pass J_k and the *current per-λ objective values* to LibMOON's EPOCore
          → get α_k ∈ ℝ^3 (Pareto-stationary mixing weights for this λ).
       e. Aggregated grad for this λ:  g_k = α_k · J_k        (shape n_var,)
  3. Average aggregated grads across the K preferences, write into policy params,
     step the optimiser.
  4. Update critic with standard MSE on per-objective returns (joint over all K).

The α from EPO is per-λ: a safety-leaning λ produces α that up-weights the
safety gradient, while a speed-leaning λ up-weights the speed gradient. This is
the key mechanism that makes the *same* network parameters serve all λ — the
network is trained on a balanced gradient signal that respects each preference.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from libmoon.solver.gradient.methods.epo_solver import EPOCore

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy
from models.critic import VectorCritic
from training.parallel_worker import collect_pref_worker


# Lower bound for any preference component before passing to EPO.
# EPO uses 1/pref internally (LP r-vector), so a true zero would blow up.
_PREF_FLOOR = 1e-3


def sample_preferences(K: int, dim: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample K preference vectors uniformly on the (dim-1)-simplex via Dirichlet(α=1).
    Returns shape (K, dim), each row sums to 1, components ≥ _PREF_FLOOR.
    """
    rng = rng or np.random.default_rng()
    prefs = rng.dirichlet(alpha=np.ones(dim), size=K).astype(np.float32)
    prefs = np.clip(prefs, _PREF_FLOOR, None)
    prefs /= prefs.sum(axis=1, keepdims=True)
    return prefs


def _write_flat_grad(params, flat: torch.Tensor) -> None:
    """Write a flat tensor back into .grad of an iterable of parameters."""
    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat[offset: offset + n].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


@dataclass
class PSLConfig:
    n_pref_samples: int = 8           # K preferences per update batch
    n_episodes_per_pref: int = 4      # episodes per λ for gradient estimation
    n_updates: int = 100              # outer loop iterations
    gamma: float = 0.99
    learning_rate: float = 3e-4
    critic_lr: float = 1e-3
    grad_clip: float = 0.5
    entropy_coef: float = 0.01        # weight on entropy bonus — prevents policy collapse
    log_interval: int = 5
    save_interval: int = 25
    checkpoint_dir: str = "checkpoints/psl"
    seed: int | None = None
    n_workers: int = 1                # parallel processes for episode collection (1 = sequential)


@dataclass
class PSLHistory:
    """Per-update training metrics for plotting/analysis."""
    update: list[int] = field(default_factory=list)
    G_safety:  list[float] = field(default_factory=list)
    G_speed:   list[float] = field(default_factory=list)
    G_comfort: list[float] = field(default_factory=list)
    L_critic:  list[float] = field(default_factory=list)
    alpha_mean: list[np.ndarray] = field(default_factory=list)   # mean EPO α across K prefs


class PSLTrainer:
    """
    Pareto Set Learning trainer using LibMOON EPO.

    Use as:
        trainer = PSLTrainer(cfg)
        trainer.train()           # full run
        # or step-by-step in a notebook:
        for step in range(cfg.n_updates):
            trainer.update_once()
    """

    def __init__(
        self,
        cfg: PSLConfig,
        env_config: Optional[dict] = None,
        policy_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = device

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        pc = policy_config or {"obs_dim": 25, "lam_dim": 3, "hidden_dim": 256, "n_actions": 5}
        self._policy_cfg = pc
        self._env_cfg    = env_config or {}

        self.policy = ConditionedPolicy(
            obs_dim=pc["obs_dim"], lam_dim=pc["lam_dim"],
            hidden_dim=pc["hidden_dim"], n_actions=pc["n_actions"],
        ).to(device)
        self.critic = VectorCritic(
            obs_dim=pc["obs_dim"], lam_dim=pc["lam_dim"],
            hidden_dim=pc["hidden_dim"],
        ).to(device)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.env = MOHighwayEnv(config=env_config)
        self.n_var   = sum(p.numel() for p in self.policy.parameters())
        self.n_c_var = sum(p.numel() for p in self.critic.parameters())
        self.history = PSLHistory()
        self._step   = 0

        self._pool: Optional[ProcessPoolExecutor] = (
            ProcessPoolExecutor(max_workers=cfg.n_workers)
            if cfg.n_workers > 1 else None
        )

    # ------------------------------------------------------------------ #
    # Core update
    # ------------------------------------------------------------------ #
    def update_once(self) -> dict:
        """
        Run one PSL update: sample K preferences, gather K×N episodes, EPO-merge
        per-objective gradients, step both policy and critic.

        When cfg.n_workers > 1 the K preference rollouts run in parallel across
        separate processes — each worker computes its own Jacobian and critic
        gradient, returning numpy arrays for aggregation here.
        """
        cfg = self.cfg
        K   = cfg.n_pref_samples

        prefs_np = sample_preferences(K, dim=3, rng=self.rng)        # (K, 3)
        prefs_t  = torch.tensor(prefs_np, dtype=torch.float32, device=self.device)
        epo_cores = [EPOCore(n_var=self.n_var, prefs=prefs_t[k:k+1]) for k in range(K)]

        # ---- collect per-preference results (parallel or sequential) ---- #
        if self._pool is not None:
            results = self._collect_parallel(prefs_np)
        else:
            results = self._collect_sequential(prefs_np)
        # results[k] = (J_k_np (3,n_var), v_k_np (3,), critic_grad_np (n_cvar,))

        # ---- EPO + gradient aggregation ---- #
        agg_policy_grad  = torch.zeros(self.n_var,   device=self.device)
        agg_critic_grad  = torch.zeros(self.n_c_var, device=self.device)
        ep_returns = []
        alphas     = []
        critic_loss_vals = []

        for k, (J_k_np, v_k_np, cg_np) in enumerate(results):
            J_k = torch.tensor(J_k_np, dtype=torch.float32, device=self.device)  # (3, n_var)
            v_k = torch.tensor(v_k_np, dtype=torch.float32, device=self.device)  # (3,)

            alpha_k = epo_cores[k].get_alpha(J_k, v_k, idx=0).to(self.device)   # (3,)
            alphas.append(alpha_k.detach().cpu().numpy())

            agg_policy_grad += alpha_k @ J_k
            agg_critic_grad += torch.tensor(cg_np, dtype=torch.float32, device=self.device)
            ep_returns.append(v_k_np)
            # Approximate critic loss for logging: mean squared norm of critic grad
            critic_loss_vals.append(float(np.mean(cg_np ** 2)))

        agg_policy_grad /= K
        agg_critic_grad /= K

        # ---------------- Policy update ---------------- #
        _write_flat_grad(self.policy.parameters(), agg_policy_grad)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=cfg.grad_clip)
        self.policy_opt.step()

        # ---------------- Critic update ---------------- #
        _write_flat_grad(self.critic.parameters(), agg_critic_grad)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=cfg.grad_clip)
        self.critic_opt.step()

        # ---------------- Logging ---------------- #
        mean_returns    = np.mean(np.stack(ep_returns), axis=0)
        mean_alpha      = np.mean(np.stack(alphas), axis=0)
        mean_critic_loss = float(np.mean(critic_loss_vals))
        self._step += 1
        self.history.update.append(self._step)
        self.history.G_safety.append(float(mean_returns[0]))
        self.history.G_speed.append(float(mean_returns[1]))
        self.history.G_comfort.append(float(mean_returns[2]))
        self.history.L_critic.append(mean_critic_loss)
        self.history.alpha_mean.append(mean_alpha)

        return {
            "step":     self._step,
            "G":        mean_returns,
            "alpha":    mean_alpha,
            "L_critic": mean_critic_loss,
        }

    def _build_worker_args(self, lam_np: np.ndarray, seed: int) -> tuple:
        policy_sd_np = {k: v.cpu().numpy() for k, v in self.policy.state_dict().items()}
        critic_sd_np = {k: v.cpu().numpy() for k, v in self.critic.state_dict().items()}
        return (
            policy_sd_np, critic_sd_np, lam_np,
            self.cfg.n_episodes_per_pref, self.cfg.gamma, self.cfg.entropy_coef,
            self._policy_cfg, self._env_cfg, seed,
        )

    def _collect_sequential(self, prefs_np: np.ndarray) -> list:
        seeds = self.rng.integers(0, 2**31, size=len(prefs_np))
        return [collect_pref_worker(self._build_worker_args(prefs_np[k], int(seeds[k])))
                for k in range(len(prefs_np))]

    def _collect_parallel(self, prefs_np: np.ndarray) -> list:
        K = len(prefs_np)
        seeds = self.rng.integers(0, 2**31, size=K)
        args = [self._build_worker_args(prefs_np[k], int(seeds[k])) for k in range(K)]
        return list(self._pool.map(collect_pref_worker, args))

    # ------------------------------------------------------------------ #
    # Full training loop
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        for step in range(1, cfg.n_updates + 1):
            info = self.update_once()
            if step % cfg.log_interval == 0:
                G = info["G"]; a = info["alpha"]
                print(
                    f"PSL upd {step:3d}/{cfg.n_updates}"
                    f"  G=[s={G[0]:.3f} v={G[1]:.3f} c={G[2]:.3f}]"
                    f"  α=[{a[0]:.2f} {a[1]:.2f} {a[2]:.2f}]"
                    f"  L_crit={info['L_critic']:.4f}"
                )
            if step % cfg.save_interval == 0:
                self.save(os.path.join(cfg.checkpoint_dir, f"psl_upd{step:04d}.pt"))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({
            "step":          self._step,
            "policy_state":  self.policy.state_dict(),
            "critic_state":  self.critic.state_dict(),
            "policy_opt":    self.policy_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        if "policy_opt" in ckpt:
            self.policy_opt.load_state_dict(ckpt["policy_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self._step = int(ckpt.get("step", 0))

    def close(self) -> None:
        self.env.close()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
