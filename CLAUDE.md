# Pareto Set Learning for Autonomous Highway Navigation

## Project Overview

Apply **LibMOON** (gradient-based multi-objective optimization) to the `highway-env` simulation to learn a **Pareto Set** of optimal driving policies. A single Preference Hypernetwork (PHN) maps a preference vector λ to optimal policy parameters, enabling dynamic adjustment of driving style (safety vs. speed vs. comfort) without retraining.

**Course:** MSML 604 — Introduction to Optimization, University of Maryland  
**Team:** Ravichandra Parvatham (122182264), Aadit Pratik Maniar (121334749), Gautam Bulusu (121968470)

---

## Optimization Formulation

Minimize a vector-valued objective over policy parameters θ:

```
min_θ  F(θ) = [f_safety(θ),  f_speed(θ),  f_comfort(θ)]ᵀ
```

| Objective | Components | Range |
|-----------|-----------|-------|
| `f_safety` | Inverse TTC with lead vehicle + min-distance violation frequency | [0, 1] |
| `f_speed` | Deviation from V_MAX (30 m/s) | [0, 1] |
| `f_comfort` | Mean absolute jerk + lateral acceleration during lane changes | [0, 1] |

All objectives are **minimized** (0 = best, 1 = worst), matching LibMOON's convention.

A **Preference Hypernetwork** H_ϕ maps a preference vector λ ∈ Λ (2-simplex) to optimal policy parameters:
```
θ* = H_ϕ(λ)        where Λ = { λ ∈ ℝ³ | Σλᵢ = 1, λᵢ ≥ 0 }
```

---

## Project Structure

```
psl/
├── envs/
│   ├── __init__.py              # exports MOHighwayEnv
│   ├── mo_highway_wrapper.py    # Multi-objective gym.Wrapper (vector reward)
│   └── objectives.py            # Pure functions: f_safety, f_speed, f_comfort
├── models/
│   ├── policy.py                # Preference-conditioned policy network π(obs, λ)
│   └── hypernetwork.py          # Preference hypernetwork H_ϕ: λ → θ (optional)
├── training/
│   ├── psl_trainer.py           # Main PSL training loop (LibMOON integration)
│   └── rollout.py               # Episode collection + per-objective return estimation
├── baselines/
│   └── scalarized_trainer.py    # Fixed-weight RL baseline for comparison
├── evaluation/
│   ├── evaluate.py              # Evaluate policy across a grid of λ values
│   └── visualize.py             # 3D Pareto surface plots
├── configs/
│   └── default.yaml             # All hyperparameters
├── train.py                     # Entry point — PSL training
├── train_baseline.py            # Entry point — baseline training
└── requirements.txt
```

---

## Environment

- **Base env:** `highway-v0` from `highway-env` (gymnasium interface)
- **Observation:** 5×5 kinematics matrix — 5 nearest vehicles × [presence, x, y, vx, vy], normalized and ego-relative
- **Actions:** Discrete(5) — `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`
- **Config overrides:** 4 lanes, 30 vehicles, 40s episodes, policy at 2 Hz

---

## Implementation Phases

### ✅ Phase 1: Environment Wrapper (`envs/`)

**Status: Complete**

Wraps `highway-env` to return a 3-vector reward instead of a scalar. Key decisions:

- `MOHighwayEnv` subclasses `gymnasium.Wrapper` — remains a fully valid Gym env
- Objectives computed from `env.unwrapped.vehicle` raw state (not the normalized obs tensor, which loses precision needed for TTC/jerk math)
- `_prev_acceleration` tracked in the wrapper for jerk computation; reset on each `env.reset()`
- `reward_space: Box(0,1, shape=(3,))` added so downstream models know output dimensionality without hardcoding
- `info["objectives"]` dict added for per-step logging

**Objective implementation details (`objectives.py`):**

- `compute_safety`: finds nearest lead vehicle in same lane → iTTC = clip(TTC_THRESHOLD / TTC, 0, 1); also checks for MIN_SAFE_DIST violation — **lane-aware only** (same lane + adjacent lane with lateral gap threshold ≤ 2m); crash hard-returns 1.0 early. Final = **weighted mean** `0.7 * iTTC + 0.3 * dist_violation` — do **not** use `max()` as the binary dist_violation dominates and kills gradient signal in the flat region.
- `compute_speed`: linear deviation from V_MAX=30; `clip((V_MAX - v) / V_MAX, 0, 1)`
- `compute_comfort`: jerk = `|Δa / dt|` normalised by MAX_JERK=5 m/s³; lat_acc = `|Δvy / dt|` (change in lateral velocity over timestep, tracked as `_prev_lateral_velocity` in the wrapper) normalised by MAX_LAT_ACC=3 m/s²; returns mean of both components. **Do not use** `|steering| * v²` — in DiscreteMetaAction, steering is a control signal, not a curvature, so that formula is dimensionally wrong. **Must divide by dt** (not just Δa) so value is frequency-independent.

---

### Phase 2: Policy + Critic (`models/`)

**Design choice — Preference-Conditioned Policy (Option A, recommended over true PHN):**

Instead of outputting full weight tensors (expensive, unstable in RL), concatenate λ directly to the observation:

```
[obs.flatten() (25-dim), λ (3-dim)] → MLP → action logits (5-dim)
```

This is equivalent to a hypernetwork when conditioning is deep enough, and is standard in multi-objective RL (MORL) literature.

**`policy.py`:**
- `ConditionedPolicy(nn.Module)`: takes `(obs, lam)` → action logits
- Architecture: `Linear(28→256) → ReLU → Linear(256→256) → ReLU → Linear(256→5)`
- `obs` is flattened from (5,5) to (25,) before concat

**`models/critic.py`** (required for A2C — replaces EMA baseline):
- `VectorCritic(nn.Module)`: takes `(obs, lam)` → value estimates `ℝ³` (one per objective)
- Architecture: `Linear(28→256) → ReLU → Linear(256→256) → ReLU → Linear(256→3)`
- Same input format as policy; output is per-objective state-value `[V_safety, V_speed, V_comfort]`
- Per-state, per-λ baselines eliminate the global-EMA-baseline flaw and the per-λ baseline mismatch problem in PSL training
- Critic loss: `L_critic = mean_i mean_t (G_i(t) - V_i(s_t, λ))²`

**`hypernetwork.py`** (optional, Phase 2b):
- True PHN: `H_ϕ: λ (3-dim) → θ` (all policy weights)
- Only implement if conditioned policy underperforms; harder to train in RL setting

---

### Phase 3: LibMOON Integration (`training/`)

LibMOON was designed for supervised/bandit settings. The bridge to RL uses **multi-objective A2C** (not REINFORCE) to get low-variance gradient estimates.

**Training loop (`psl_trainer.py`):**
1. Sample batch of K preference vectors `{λ_1, ..., λ_K}` uniformly from the 2-simplex
2. For each λ_k: run N episodes using `π(obs, λ_k)`, collect per-step costs, log-probs, and critic values `V_i(s_t, λ_k)`
3. Compute per-objective advantages: `A_i(t) = G_i(t) - V_i(s_t, λ_k)` (critic provides per-state, per-λ baseline)
4. For each objective i, compute A2C policy loss: `L_i = -mean_t [ A_i(t) * log π(a_t | s_t, λ_k) ]`
5. Call `L_i.backward(retain_graph=True)`, extract `∇_ϕ L_i`, zero grad — repeat for all 3 objectives
6. Stack into K×3 objective matrix and K×3 gradient tensors
7. Pass to LibMOON's EPO solver → aggregated gradient → update ϕ
8. Also update critic by minimizing `L_critic = mean_i mean_t (G_i(t) - V_i(s_t, λ_k))²`

**`rollout.py`:**
- Runs one episode, returns per-step `reward_vec`, log-probs, and critic value estimates
- Computes discounted returns per objective: `G_i = Σ γᵗ r_i(t)`
- Returns `(returns (T,3), log_probs (T,), values (T,3))`

**LibMOON bridge — how to pass RL gradients to EPO:**
LibMOON's `EPO` solver accepts gradient vectors directly; it does not need autograd into the loss itself. The correct pattern:
```python
grads = []   # list of K lists, each list has 3 gradient vectors
for i in range(3):
    loss_i = -(advantages[:, i] * log_probs).mean()
    policy.zero_grad()
    loss_i.backward(retain_graph=True)
    grads_i = torch.cat([p.grad.flatten() for p in policy.parameters()])
    grads.append(grads_i)
# EPO returns scalar weights w_i per objective
weights = epo_solver.get_weighted_loss(obj_values, grads)
# Apply: combined_loss = sum(w_i * loss_i); combined_loss.backward(); optimizer.step()
```

**Key gotcha on `n_episodes_per_pref`:** Use at least 10–20 episodes per λ per update (not 3). With 80-step episodes and high environment stochasticity, 3 episodes produce extremely noisy gradient estimates that destabilize EPO weight computation. The critic reduces variance substantially, but episode count still matters.

---

### Phase 4: Baseline (`baselines/`)

**`scalarized_trainer.py`:** Standard PPO or REINFORCE with scalar reward:
```
r_scalar = λ_fixed · [r_safety, r_speed, r_comfort]
```

Train one agent per fixed λ at corners and center of simplex:
- `λ = (1, 0, 0)` — safety only
- `λ = (0, 1, 0)` — speed only
- `λ = (0, 0, 1)` — comfort only
- `λ = (1/3, 1/3, 1/3)` — uniform

**Comparison metrics:**
- Each fixed-λ baseline vs. PSL policy evaluated at that same λ
- Pareto coverage: does PSL approximate the full front, or just discrete points?
- Sample efficiency: training steps to reach equivalent per-objective performance

---

### Phase 5: Evaluation & Visualization (`evaluation/`)

**`evaluate.py`:**
- Load trained PSL policy
- Evaluate at a uniform grid of ~100 λ values on the 2-simplex (use `np.random.dirichlet` or a triangular grid)
- Record mean episode returns `[G_safety, G_speed, G_comfort]` per λ
- Save results as CSV

**`visualize.py`:**
- **3D Pareto surface:** `matplotlib` `plot_trisurf` with objectives on each axis
- **2D trade-off projections:** 3 pairwise plots (safety vs. speed, safety vs. comfort, speed vs. comfort)
- Overlay baseline points on same plots

---

## Implementation Order

| Step | What | Why first |
|------|------|-----------|
| 1 ✅ | Install deps, verify `highway-env` | De-risk env setup |
| 2 ✅ | `mo_highway_wrapper.py` + `objectives.py` | Everything depends on correct objectives |
| 3 | `policy.py` + `critic.py` — conditioned policy + vector critic, train with **single fixed λ using A2C** | Verify RL loop works before adding MOO |
| 4 | `psl_trainer.py` — LibMOON integration | Core contribution |
| 5 | `scalarized_trainer.py` — baseline | Needed for comparison |
| 6 | `evaluate.py` + `visualize.py` | Final results |

---

## Environment Setup

```bash
# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install highway-env gymnasium torch numpy pyyaml libmoon matplotlib
```

All commands should be run with `.venv/bin/python3` or after activating the venv.

---

## Key Constants (tunable in `configs/default.yaml`)

| Constant | Value | Used in |
|----------|-------|---------|
| `V_MAX` | 30.0 m/s | `f_speed` |
| `TTC_THRESHOLD` | 5.0 s | `f_safety` |
| `MIN_SAFE_DIST` | 10.0 m | `f_safety` |
| `MAX_JERK` | 5.0 m/s³ | `f_comfort` |
| `MAX_LAT_ACC` | 3.0 m/s² | `f_comfort` |
| `policy_frequency` | 2 Hz → dt=0.5s | wrapper + jerk |
| `lanes_count` | 4 | env config |
| `vehicles_count` | 30 | env config |
| `duration` | 40 s | env config |

---

## Notes & Gotchas

- **`ego.history` is always empty** in `MDPVehicle` — do not use it. Track `_prev_acceleration` and `_prev_lateral_velocity` manually in the wrapper; both reset on `env.reset()`.
- **Reach into `env.unwrapped`** for raw vehicle state (TTC, jerk). The normalized obs tensor doesn't have enough precision.
- **LibMOON expects minimization** — all objectives must be costs (lower = better), not rewards.
- **Jerk must be divided by dt** — `|Δa / dt|`, not just `|Δa|`, otherwise value depends on `policy_frequency`.
- **Lateral acceleration must use `Δvy / dt`**, not `|steering| * v²`. In DiscreteMetaAction, `ego.action["steering"]` is a control signal, not a curvature — the centripetal formula `κv²` is wrong here. Track `_prev_lateral_velocity = ego.velocity[1]` each step and compute `|Δvy / dt|`.
- **`dist_violation` must be lane-aware** — checking Euclidean distance across all lanes fires constantly on a 4-lane highway (adjacent-lane cars are 4–8m away, well within MIN_SAFE_DIST=10m). Only flag vehicles in the same or immediately adjacent lane, with a lateral gap threshold (≤ 2m).
- **Do not use `max()` to combine iTTC and dist_violation** — the binary dist_violation dominates and creates a flat gradient landscape. Use `0.7 * iTTC + 0.3 * dist_violation` for a smooth, differentiable signal.
- **Global EMA baseline does not work across λ values** — different preferences produce different expected returns. Use the λ-conditioned critic `V(obs, λ)` as the baseline in all training phases.
- **`normalize_reward: False`** in env config — we bypass the built-in scalar reward entirely.
- The pygame `pkg_resources` deprecation warning is harmless; suppress with `PYTHONWARNINGS=ignore` if needed.
