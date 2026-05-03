# Pareto Set Learning for Autonomous Highway Navigation

A single neural policy that drives in different "personalities" — safety-first,
speed-first, comfort-first, or anything in between — by simply changing a
3-element preference vector at inference time. **No retraining required.**

Built on top of [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv)
and [`LibMOON`](https://github.com/xzhang2523/libmoon).

> **Course:** MSML 604 — Introduction to Optimization, University of Maryland
> **Team:** Ravichandra Parvatham, Aadit Pratik Maniar, Gautam Bulusu

---

## Problem Statement

An autonomous vehicle must continuously trade off three conflicting goals:

| Objective | What it measures | Range |
|-----------|------------------|-------|
| `f_safety`  | Inverse time-to-collision + min-distance violations | [0, 1] |
| `f_speed`   | Deviation from the speed limit (30 m/s) | [0, 1] |
| `f_comfort` | Mean absolute jerk + lateral acceleration | [0, 1] |

All three are **costs** (lower = better). Classical RL solves this by
hand-picking a fixed weighting `r = λ₁·r_safety + λ₂·r_speed + λ₃·r_comfort` and
training a separate agent for each `λ`. That doesn't scale: the right `λ`
depends on weather, traffic, passenger preference, and time of day.

We want a *single* model that exposes the entire **Pareto front** —
the set of policies where no objective can improve without another getting
worse — and lets the user pick a point on that front at runtime.

---

## Solution Logic

### 1. Multi-objective environment ([envs/](envs/))

`MOHighwayEnv` wraps `highway-v0` and replaces the scalar reward with a 3-vector
of costs computed from raw simulator state (TTC, jerk, lateral acceleration).

### 2. Preference-conditioned policy ([models/](models/))

Instead of a separate network per `λ`, we **concatenate λ to the observation**
before the MLP:

```
[obs.flatten() (25-d), λ (3-d)]  →  MLP  →  action logits (5-d)
```

A matching `VectorCritic` outputs **per-objective** value estimates
`V(s, λ) ∈ ℝ³` so each objective gets its own per-state baseline.

### 3. Pareto Set Learning loop ([training/psl_trainer.py](training/psl_trainer.py))

Each PSL update:

1. Sample `K` preferences `λ_1…λ_K` from a Dirichlet over the 2-simplex.
2. For each `λ_k`, run `N` episodes; collect per-step costs, log-probs, and
   critic values.
3. Build **three separate per-objective policy losses**
   `L_i = mean_t [ A_i(t) · log π(a_t | s_t, λ_k) ]`
   and run three backward passes to get a Jacobian
   `J_k ∈ ℝ^(3 × n_params)`.
4. Pass `J_k` and the current per-objective costs to **LibMOON's EPO solver** →
   it returns mixing weights `α_k ∈ ℝ³` that yield a Pareto-stationary descent
   direction matching `λ_k`.
5. Apply the merged gradient `α_k · J_k` to the policy. Average across the K
   preferences before stepping the optimiser.
6. Update the critic with standard MSE on the per-objective returns.

The key insight: EPO doesn't just scalarise with `λ`. It computes the descent
direction that *moves objectives toward the Pareto-optimal ratio* implied by
`λ`, given where the policy currently sits in objective space. That's why one
network can serve all preferences.

### 4. Evaluation ([evaluation/](evaluation/))

After training, we evaluate the **frozen** policy at a triangular grid of
`λ` values on the 3-simplex. Each grid point is one query at inference time,
producing the corresponding driving style. Plotting the resulting cost
vectors traces out the learned Pareto surface.

---

## Project Layout

```
autonomous-driving-psl/
├── envs/                    # MOHighwayEnv + objective functions
├── models/                  # ConditionedPolicy, VectorCritic
├── training/
│   ├── rollout.py           # episode collection
│   └── psl_trainer.py       # PSL training loop with LibMOON EPO
├── baselines/
│   └── scalarized_trainer.py # fixed-λ A2C baseline
├── evaluation/
│   ├── evaluate.py          # single-λ + grid evaluation
│   └── visualize.py         # Pareto surface plots
├── configs/default.yaml
├── train.py                 # single-λ entry point
├── explore.ipynb            # interactive walk-through (recommended)
├── requirements.txt
└── CLAUDE.md                # design notes
```

---

## Installation

```bash
# 1. Create venv with uv (or python -m venv .venv)
uv venv .venv
source .venv/bin/activate

# 2. Install everything in requirements.txt EXCEPT libmoon
.venv/bin/pip install highway-env gymnasium torch numpy pyyaml matplotlib \
    seaborn cvxpy cvxopt opencv-python pymoo tqdm scikit-learn pandas \
    jupyter ipykernel

# 3. Install libmoon from source WITHOUT its dependencies (its torch==2.2.1
#    pin conflicts with newer Python). Its actual runtime deps are already
#    covered by step 2.
.venv/bin/pip install --no-deps git+https://github.com/xzhang2523/libmoon.git
```

Verify:
```bash
.venv/bin/python3 -c "from libmoon.solver.gradient.methods.epo_solver import EPOCore; print('libmoon OK')"
.venv/bin/python3 -c "import highway_env, gymnasium; print('env OK')"
```

---

## Running the Code

### Interactive notebook (recommended)

```bash
.venv/bin/jupyter lab explore.ipynb
```

Walks through every layer end to end:

| § | What |
|---|------|
| 1 | Environment wrapper — see raw observations and the 3-vector cost |
| 2 | Objective functions — sanity-check curves for `f_speed`, jerk, iTTC |
| 3 | Policy + critic — parameter counts, action probabilities at different `λ` |
| 4 | Rollout collection — episode tensors and per-objective discounted returns |
| 5 | Advantage signal — `A(t) = G(t) − V(s, λ)` |
| 6 | Mini A2C loop at fixed `λ` (verifies RL plumbing) |
| 7 | **Pareto Set Learning with LibMOON EPO** |
| 8 | **Adaptation demo** — same network, three different λ, three driving styles |
| 9 | Pareto surface — full simplex sweep with 3D and 2D projections |

### Command-line training

```bash
# Single-λ A2C baseline at one preset (one of: safety, speed, comfort, uniform)
.venv/bin/python3 baselines/scalarized_trainer.py --lam uniform

# All four presets sequentially
.venv/bin/python3 baselines/scalarized_trainer.py --all
```

PSL training is exposed via the trainer class — drive it from a script or the
notebook:

```python
from training.psl_trainer import PSLTrainer, PSLConfig

trainer = PSLTrainer(PSLConfig(n_pref_samples=8, n_episodes_per_pref=4, n_updates=200))
trainer.train()
trainer.save("checkpoints/psl/final.pt")
```

### Evaluation

```python
from evaluation import evaluate_policy_at_lambda, evaluate_policy_grid, plot_pareto_3d

# Same network, different driving style — no retraining
out = evaluate_policy_at_lambda(trainer.policy, [1.0, 0.0, 0.0], n_episodes=5)
print(out["mean"])    # [low f_safety, higher f_speed, ?]

# Sweep the simplex
grid = evaluate_policy_grid(trainer.policy, n_per_side=6, n_episodes=3)
plot_pareto_3d(grid)
```

---

## Hyperparameters

All in [configs/default.yaml](configs/default.yaml). Key knobs:

| Key | Meaning | Default |
|-----|---------|---------|
| `train.n_episodes` | episodes for the single-λ baseline | 500 |
| `train.gamma` | discount factor | 0.99 |
| `train.learning_rate` / `train.critic_lr` | Adam LRs | 3e-4 / 1e-3 |
| `psl.n_pref_samples` (`K`) | preferences per PSL update | 8 |
| `psl.n_episodes_per_pref` (`N`) | episodes per preference (≥10 recommended) | 15 |
| `psl.solver` | LibMOON solver | `epo` |
| `eval.n_lambda_points` | grid size for Pareto surface eval | 100 |

---

## Notes / Gotchas

- LibMOON expects **minimisation** — every reward signal flows through the
  pipeline as a cost.
- The pygame `pkg_resources` deprecation warning is harmless; suppress it with
  `PYTHONWARNINGS=ignore`.
- The cv2 / pygame `objc` SDL2 collision warnings on macOS are also harmless —
  both ship the same `libSDL2.dylib`.
- See [CLAUDE.md](CLAUDE.md) for deeper design notes (objective implementation
  details, why we use weighted-mean instead of `max()` for safety, etc).
