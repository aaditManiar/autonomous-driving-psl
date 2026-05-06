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
import warnings
warnings.filterwarnings("ignore")
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


def sample_preferences(
    K: int,
    dim: int = 3,
    rng: np.random.Generator | None = None,
    p_corner: float = 0.0,
) -> np.ndarray:
    """
    Sample K preference vectors on the (dim-1)-simplex via Dirichlet(α=1).
    Returns shape (K, dim), each row sums to 1, components ≥ _PREF_FLOOR.

    With p_corner > 0, replace each sample with a near-corner λ
    (one component ≈ 1-ε, others ≈ ε) with that probability, picking
    the corner uniformly among the `dim` axes. This guards against
    the policy collapsing on the simplex interior because Dirichlet(1)
    almost never produces samples near the corners.
    """
    rng = rng or np.random.default_rng()
    prefs = rng.dirichlet(alpha=np.ones(dim), size=K).astype(np.float32)

    if p_corner > 0.0:
        replace_mask = rng.random(K) < p_corner
        n_replace = int(replace_mask.sum())
        if n_replace > 0:
            corner_axes = rng.integers(0, dim, size=n_replace)
            corners = np.full((n_replace, dim), _PREF_FLOOR, dtype=np.float32)
            corners[np.arange(n_replace), corner_axes] = 1.0 - _PREF_FLOOR * (dim - 1)
            prefs[replace_mask] = corners

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
    gamma: float = 0.97
    learning_rate: float = 3e-4
    critic_lr: float = 1e-3
    grad_clip: float = 0.5
    entropy_coef: float = 0.01        # weight on entropy bonus — prevents policy collapse
    p_corner: float = 0.25            # fraction of sampled λ replaced with near-corner one-hot — forces the policy to see extreme preferences
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
    L_critic:  list[float] = field(default_factory=list)         # mean MSE of (returns − values)
    L_policy_safety:  list[float] = field(default_factory=list)  # per-objective policy loss
    L_policy_speed:   list[float] = field(default_factory=list)
    L_policy_comfort: list[float] = field(default_factory=list)
    alpha_mean:  list[np.ndarray] = field(default_factory=list)  # mean EPO α across K prefs
    crash_rate:  list[float]      = field(default_factory=list)  # fraction of episodes that crashed


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

        self._last_crash_rate: float = 0.0   # used by entropy annealing on the next update
        self._crash_rate_ema:  float = 0.5   # exponential moving average — initialised high so
                                              # the α floor starts conservative on fresh/resumed runs

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

        # Entropy annealing: when crash rate was high last update, reduce exploration so the
        # training distribution doesn't stay crash-dominated (which confuses the critic).
        # Floor at 20% of base so entropy never collapses entirely.
        self._effective_entropy = cfg.entropy_coef * max(0.2, 1.0 - 1.5 * self._last_crash_rate)

        prefs_np = sample_preferences(K, dim=3, rng=self.rng, p_corner=cfg.p_corner)  # (K, 3)
        prefs_t  = torch.tensor(prefs_np, dtype=torch.float32, device=self.device)
        epo_cores = [EPOCore(n_var=self.n_var, prefs=prefs_t[k:k+1]) for k in range(K)]

        # ---- collect per-preference results (parallel or sequential) ---- #
        if self._pool is not None:
            results = self._collect_parallel(prefs_np)
        else:
            results = self._collect_sequential(prefs_np)
        # results[k] = (J_k_np (3,n_var), v_k_np (3,), critic_grad_np (n_cvar,),
        #               policy_loss_vec_np (3,), critic_loss_scalar)

        # ---- EPO + gradient aggregation ---- #
        agg_policy_grad  = torch.zeros(self.n_var,   device=self.device)
        agg_critic_grad  = torch.zeros(self.n_c_var, device=self.device)
        ep_returns       = []
        alphas           = []
        policy_loss_vecs = []
        critic_loss_vals = []
        crash_rates      = []

        for k, (J_k_np, v_k_np, cg_np, pl_vec_np, cl_scalar, cr) in enumerate(results):
            J_k = torch.tensor(J_k_np, dtype=torch.float32, device=self.device)  # (3, n_var)
            v_k = torch.tensor(v_k_np, dtype=torch.float32, device=self.device)  # (3,)

            # Normalise each Jacobian row to unit norm before passing to EPO.
            # Without this, the safety row (nonzero iTTC everywhere after the formula fix)
            # has consistently larger magnitude than speed/comfort rows, causing EPO's LP
            # to treat safety as always dominant and return α≈[1,0,0] regardless of λ.
            row_norms = J_k.norm(dim=1, keepdim=True).clamp(min=1e-8)
            J_k_epo   = J_k / row_norms

            alpha_k = epo_cores[k].get_alpha(J_k_epo, v_k, idx=0).to(self.device)   # (3,)

            # Enforce a minimum safety weight using a smoothed (EMA) crash rate.
            # Using raw _last_crash_rate was too volatile — one lucky low-crash batch
            # dropped the floor to 0.05, letting speed dominate for 30 updates and
            # spiking crash rate from 11% back to 41% (observed upd 210→260).
            # EMA decays at 30%/update so the floor retains memory of recent crashes.
            min_alpha_safety = max(0.10, self._crash_rate_ema * 0.5)
            if alpha_k[0].item() < min_alpha_safety:
                alpha_k = alpha_k.clone()
                alpha_k[0] = torch.tensor(min_alpha_safety, dtype=torch.float32, device=self.device)
                alpha_k = alpha_k / alpha_k.sum()

            alphas.append(alpha_k.detach().cpu().numpy())

            agg_policy_grad += alpha_k @ J_k
            agg_critic_grad += torch.tensor(cg_np, dtype=torch.float32, device=self.device)
            ep_returns.append(v_k_np)
            policy_loss_vecs.append(pl_vec_np)
            critic_loss_vals.append(cl_scalar)
            crash_rates.append(cr)

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
        mean_returns     = np.mean(np.stack(ep_returns), axis=0)
        mean_alpha       = np.mean(np.stack(alphas), axis=0)
        mean_policy_loss = np.mean(np.stack(policy_loss_vecs), axis=0)   # (3,)
        mean_critic_loss = float(np.mean(critic_loss_vals))
        mean_crash_rate  = float(np.mean(crash_rates))
        self._last_crash_rate = mean_crash_rate                                      # raw — for entropy annealing
        self._crash_rate_ema  = 0.7 * self._crash_rate_ema + 0.3 * mean_crash_rate  # smoothed — for α floor
        self._step += 1
        self.history.update.append(self._step)
        self.history.G_safety.append(float(mean_returns[0]))
        self.history.G_speed.append(float(mean_returns[1]))
        self.history.G_comfort.append(float(mean_returns[2]))
        self.history.L_critic.append(mean_critic_loss)
        self.history.L_policy_safety.append(float(mean_policy_loss[0]))
        self.history.L_policy_speed.append(float(mean_policy_loss[1]))
        self.history.L_policy_comfort.append(float(mean_policy_loss[2]))
        self.history.alpha_mean.append(mean_alpha)
        self.history.crash_rate.append(mean_crash_rate)

        return {
            "step":        self._step,
            "G":           mean_returns,
            "alpha":       mean_alpha,
            "L_critic":    mean_critic_loss,
            "L_policy":    mean_policy_loss,
            "crash_rate":  mean_crash_rate,
            "entropy_coef": self._effective_entropy,
        }

    def _build_worker_args(self, lam_np: np.ndarray, seed: int) -> tuple:
        policy_sd_np = {k: v.cpu().numpy() for k, v in self.policy.state_dict().items()}
        critic_sd_np = {k: v.cpu().numpy() for k, v in self.critic.state_dict().items()}
        return (
            policy_sd_np, critic_sd_np, lam_np,
            self.cfg.n_episodes_per_pref, self.cfg.gamma, self._effective_entropy,
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
                G = info["G"]; a = info["alpha"]; lp = info["L_policy"]
                print(
                    f"PSL upd {step:3d}/{cfg.n_updates}"
                    f"  G=[s={G[0]:.3f} v={G[1]:.3f} c={G[2]:.3f}]"
                    f"  α=[{a[0]:.2f} {a[1]:.2f} {a[2]:.2f}]"
                    f"  L_pol=[{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}]"
                    f"  L_crit={info['L_critic']:.4f}"
                    f"  crash={info['crash_rate']:.1%}"
                    f"  entr={info['entropy_coef']:.4f}"
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
