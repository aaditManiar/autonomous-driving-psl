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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from libmoon.solver.gradient.methods.epo_solver import EPOCore

from envs import MOHighwayEnv
from models.policy import ConditionedPolicy
from models.critic import VectorCritic
from training.rollout import collect_episode, episode_summary


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


def _flat_grad(params) -> torch.Tensor:
    """Concatenate .grad of an iterable of parameters into one flat tensor."""
    return torch.cat([
        (p.grad.detach().flatten() if p.grad is not None
         else torch.zeros_like(p).flatten())
        for p in params
    ])


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
        self.n_var = sum(p.numel() for p in self.policy.parameters())
        self.history = PSLHistory()
        self._step = 0

    # ------------------------------------------------------------------ #
    # Core update
    # ------------------------------------------------------------------ #
    def update_once(self) -> dict:
        """
        Run one PSL update: sample K preferences, gather K×N episodes, EPO-merge
        per-objective gradients, step both policy and critic.

        Returns a dict of per-update metrics.
        """
        cfg = self.cfg
        K = cfg.n_pref_samples
        N = cfg.n_episodes_per_pref

        prefs_np = sample_preferences(K, dim=3, rng=self.rng)         # (K, 3)
        prefs_t  = torch.tensor(prefs_np, dtype=torch.float32, device=self.device)

        # One EPOCore per preference (each LP is parameterised by 1/pref).
        epo_cores = [EPOCore(n_var=self.n_var, prefs=prefs_t[k:k + 1]) for k in range(K)]

        # Accumulators across K preferences.
        agg_grad   = torch.zeros(self.n_var, device=self.device)
        ep_returns = []       # list of (3,) arrays — final cumulative cost vectors
        critic_losses = []
        alphas = []

        for k in range(K):
            lam_np = prefs_np[k]
            jacobian_rows: list[torch.Tensor] = []
            obj_values:    list[float] = []
            losses_for_critic: list[torch.Tensor] = []

            for _ in range(N):
                returns, log_probs, values, entropies = collect_episode(
                    self.env, self.policy, self.critic, lam_np,
                    gamma=cfg.gamma, device=self.device,
                )
                advantages = (returns - values).detach()             # (T, 3)

                T = log_probs.shape[0]
                weighted = advantages * log_probs.unsqueeze(1)       # (T, 3)
                # Entropy bonus: subtract entropy from each objective's loss so that
                # gradient descent also pushes the policy toward higher entropy
                # (more exploration). The same entropy term appears in all 3 rows of
                # the Jacobian, so EPO mixes it with net coefficient Σα_i = 1.
                per_obj = weighted.mean(dim=0) - cfg.entropy_coef * entropies.mean()  # (3,)

                # Per-objective values for EPO: use mean per-step cost so values
                # stay in [0, 1] and EPO's LP is properly calibrated across objectives.
                # Recover per-step costs from consecutive discounted returns:
                #   r[t] = G[t] - γ·G[t+1]  (valid for t < T-1)
                if T > 1:
                    returns_np = returns.detach().cpu().numpy()      # (T, 3)
                    step_costs_np = returns_np[:-1] - cfg.gamma * returns_np[1:]  # (T-1, 3)
                    mean_step_cost = step_costs_np.mean(axis=0)      # (3,) in [0, 1]
                else:
                    mean_step_cost = returns[0].detach().cpu().numpy()
                obj_values.append(mean_step_cost)

                # Critic MSE (joint across objectives and timesteps).
                losses_for_critic.append(((returns - values) ** 2).mean())

                # Compute Jacobian for this episode by 3 backward passes.
                ep_jac = torch.zeros(3, self.n_var, device=self.device)
                for i in range(3):
                    self.policy.zero_grad(set_to_none=False)
                    retain = i < 2
                    per_obj[i].backward(retain_graph=retain)
                    ep_jac[i] = _flat_grad(self.policy.parameters())
                jacobian_rows.append(ep_jac)

            # Average Jacobian and objective values across the N episodes for this λ.
            J_k = torch.stack(jacobian_rows, dim=0).mean(dim=0)              # (3, n_var)
            v_k = torch.tensor(np.mean(obj_values, axis=0), dtype=torch.float32)  # (3,)

            # EPO mixing weights for this preference.
            alpha_k = epo_cores[k].get_alpha(J_k, v_k, idx=0).to(J_k.device)  # (3,)
            alphas.append(alpha_k.detach().cpu().numpy())

            # Aggregated gradient for this λ.
            g_k = alpha_k @ J_k                                              # (n_var,)
            agg_grad = agg_grad + g_k

            ep_returns.append(v_k.numpy())
            critic_losses.append(torch.stack(losses_for_critic).mean())

        agg_grad = agg_grad / K

        # ---------------- Policy update ---------------- #
        _write_flat_grad(self.policy.parameters(), agg_grad)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=cfg.grad_clip)
        self.policy_opt.step()

        # ---------------- Critic update ---------------- #
        self.critic_opt.zero_grad()
        L_critic_total = torch.stack(critic_losses).mean()
        L_critic_total.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=cfg.grad_clip)
        self.critic_opt.step()

        # ---------------- Logging ---------------- #
        mean_returns = np.mean(np.stack(ep_returns), axis=0)
        mean_alpha   = np.mean(np.stack(alphas), axis=0)
        self._step += 1
        self.history.update.append(self._step)
        self.history.G_safety.append(float(mean_returns[0]))
        self.history.G_speed.append(float(mean_returns[1]))
        self.history.G_comfort.append(float(mean_returns[2]))
        self.history.L_critic.append(float(L_critic_total.item()))
        self.history.alpha_mean.append(mean_alpha)

        return {
            "step":      self._step,
            "G":         mean_returns,
            "alpha":     mean_alpha,
            "L_critic":  float(L_critic_total.item()),
        }

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
