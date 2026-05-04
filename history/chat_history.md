# Chat History — PSL Autonomous Driving Project

Compiled record of the working session for the MSML 604 PSL highway-env project.

---

## 1. Git: pushing to a new GitHub repo

**Q:** The original remote repo is being deleted. How do I push everything to a new repo from scratch?

**A:**
1. Create a new repo on GitHub (no README/gitignore).
2. Run:
   ```bash
   git remote remove origin
   git remote add origin https://github.com/USER/repo.git
   git push -u origin main
   git push --tags
   ```

### Sub-questions

- **What does `-u` do in `git push`?** Sets upstream tracking, so future bare `git push` / `git pull` commands know where to go.
- **What does `git push --tags` do?** Pushes local tags (e.g. `v1.0`). Tags aren't pushed automatically.

---

## 2. Resetting commit history

**Q:** How to reset the commit history.

**A:** Use an orphan branch:

```bash
git checkout --orphan fresh-start
git add -A
git commit -m "initial commit"
git branch -D main
git branch -m main
git push -u origin main --force
```

**Q:** `rm -r .git` is asking for confirmation — how do I delete and reinitialise?

**A:** Use `rm -rf .git`, then `git init`, `git add -A`, `git commit -m "initial commit"`, add remote, force-push.

---

## 3. Untracking CLAUDE.md from the remote

**Q:** Remove `CLAUDE.md` from the remote but keep it locally; ensure it's not pushed in the future.

**A:**
```bash
echo "CLAUDE.md" >> .gitignore
git rm --cached CLAUDE.md
git add .gitignore
git commit -m "untrack CLAUDE.md"
git push
```

---

## 4. Listing files unused by `explore.ipynb`

**Q:** List files not used (directly or transitively) by the explore notebook.

**A:** Traced the import closure starting from notebook cells.

**NOT in the closure:**
- `baselines/__init__.py`
- `baselines/scalarized_trainer.py`
- `train.py`
- `README.md`
- `CLAUDE.md`
- `requirements.txt`

**IN the closure:** `envs/__init__.py`, `envs/mo_highway_wrapper.py`, `envs/objectives.py`, `models/__init__.py`, `models/policy.py`, `models/critic.py`, `training/__init__.py`, `training/rollout.py`, `training/psl_trainer.py`, `evaluation/__init__.py`, `evaluation/evaluate.py`, `evaluation/visualize.py`, `configs/default.yaml`.

---

## 5. PSL trainer is independent of mini-A2C

**Q:** Does the PSL loop depend on the mini A2C training loop in the notebook?

**A:** No. The mini A2C cell creates its own `policy_fresh`/`critic_fresh` and trains them on a fixed λ. The PSL section instantiates `PSLTrainer` which builds its own policy, critic, and env. You could skip the mini A2C cells entirely and the PSL section would still work (provided imports from cell 1 are run).

---

## 6. PSL config tuning (initial)

**Q:** Call PSL trainer with appropriate config to view desirable results.

**A:** Updated PSL config cell:
- `n_pref_samples=6` (was 4) — better simplex coverage
- `n_episodes_per_pref=12` (was 2) — meets CLAUDE.md gotcha (≥10 to avoid noisy gradients)
- `n_updates=400` (was 230) — long enough for surface to separate
- `entropy_coef=0.01` explicit — prevent policy collapse

---

## 7. Notebook update for `collect_episode` signature change

**Q:** ValueError: too many values to unpack (expected 3) — `collect_episode` now returns 4 values; need to incorporate entropies.

**A:** Updated three cells:
- Section 4 rollout cell: unpack `entropies`, print mean entropy with `ln(5)` reference.
- Section 6 mini-A2C loop: unpack `ents`, add `- ENTROPY_COEF * ents.mean()` to policy loss, track entropy in history.
- Section 6 plot cell: switched to 2×3 grid, added "Mean policy entropy" panel.

---

## 8. Explaining a PSL training log line

**Log line:**
```
PSL upd 183/200  G=[s=0.164 v=0.034 c=0.098]  α=[0.46 0.20 0.34]  L_crit=0.3468
```

### G — per-step cost vector

`[G_safety, G_speed, G_comfort]` averaged over K preferences × N episodes. Each in `[0, 1]`. Lower is better. Computed by unrolling discounted returns into per-step costs.

### α — EPO mixing weights

Convex combination (sums to 1) used to merge per-objective gradients into one descent direction:
```
g_combined = α_safety · ∇L_safety + α_speed · ∇L_speed + α_comfort · ∇L_comfort
```

**α ≠ λ.** λ is the user's preference. α is what EPO computes given current losses, gradient geometry, and λ. EPO concentrates α on whichever objective is currently the bottleneck.

---

## 9. Why does safety-only λ produce *worse* safety than other rows?

**Eval result:**
```
safety-only  → safety=0.214  speed=0.176  comfort=0.036
speed-only   → safety=0.085  speed=0.171  comfort=0.013
comfort-only → safety=0.086  speed=0.172  comfort=0.014
uniform      → safety=0.081  speed=0.172  comfort=0.012
```

**A:** Real failure mode. Three reasons:

1. **Training never visits simplex corners.** `Dirichlet(α=1)` produces samples in the interior; expected min-coordinate ≈ 0.07. Eval at `(0.99, 0.005, 0.005)` is therefore extrapolation outside the training distribution.
2. **Other three rows look identical** — confirms the policy collapsed onto a single near-default behaviour and barely conditions on λ. The corner row diverges in the wrong direction because the policy has no idea what to do there.
3. **Fixes in priority order:** corner-sample injection, `Dirichlet(α=0.3)`, sanity-check policy responds to λ at all, more updates after #1.

---

## 10. Adding corner sampling

**Q:** Introduce corner sampling.

**A:** Modified [training/psl_trainer.py](training/psl_trainer.py):
- Extended `sample_preferences` with `p_corner` arg. With probability `p_corner`, replaces a Dirichlet sample with a near-corner one-hot (one component ≈ 1, others ≈ ε).
- Added `p_corner: float = 0.25` to `PSLConfig`.
- Passed `cfg.p_corner` through to the sampler at every update.
- Updated notebook PSL config cell to make `p_corner` visible.

---

## 11. Why do I need 400 updates if `L_crit` drops to 0 in 50?

**A:** `L_crit ≈ 0` means the critic has caught up to the *current* policy's value function. They optimise separately:
- Critic loss is supervised regression on a near-stationary target — converges fast.
- Policy improvement is driven by tiny gradient signals from advantages, fighting noisy A2C estimates — much slower.

**A flat `L_crit` near 0 is a *symptom* that the policy isn't moving from update to update.** The right convergence signals are: per-objective `G[s,v,c]` trending down, EPO `α` distribution stabilising, behavioural separation across λ.

In our regime: `L_crit ≈ 0`, `G` not changing much, policy collapsed onto single near-default behaviour. **400 vs 50 updates won't fix that** — the corner sampling change is the actual fix.

---

## 12. Bugs in PSL trainer / reward formulation

**Q:** Is there any issue with the PSL trainer? Why is policy loss not being logged with critic loss?

### Bug 1: `L_crit` mislabel

[training/psl_trainer.py:227](training/psl_trainer.py#L227) was logging:
```python
critic_loss_vals.append(float(np.mean(cg_np ** 2)))   # ← mean(grad²), NOT loss
```

The actual MSE was computed in the worker but never returned to the main process. So `L_crit` "dropping to ≈ 0" was actually the gradient norm shrinking, not the loss converging.

### Bug 2: Policy loss never logged

[training/parallel_worker.py](training/parallel_worker.py) computed `per_obj` (3 scalar policy losses) but the return tuple dropped them. Main process had no way to log `L_policy` at all.

### Reward / loss sign convention (correct)

For costs (minimise) instead of rewards, the policy loss is the opposite sign of textbook A2C:
```python
weighted = advantages * log_probs.unsqueeze(1)
per_obj  = weighted.mean(dim=0) - entropy_coef * entropies.mean()
```
When `A > 0`, action was *worse* than expected → minimising loss pushes log π **down**. Entropy term subtracted because we want to maximise entropy.

### Fix applied

- `parallel_worker.py`: returns 5-tuple now: `(J_k, v_k, critic_grad, policy_loss_vec, critic_loss_scalar)`.
- `psl_trainer.py`: `PSLHistory` extended with `L_policy_safety`, `L_policy_speed`, `L_policy_comfort`. `update_once()` consumes the new fields, replaces the bogus metric with real critic MSE. `train()` print line now shows `L_pol=[...]` alongside `L_crit=...`.
- Notebook section 7.3 plot cell switched to 2×2 grid: per-objective `G`, EPO `α`, per-objective `L_policy`, corrected `L_critic`.

---

## 13. Reading a healthier training log

```
PSL upd 78/150  G=[s=0.189 v=0.018 c=0.134]  α=[0.40 0.12 0.48]  L_pol=[-0.060 -0.018 -0.091]  L_crit=0.4013
```

**Healthy signs:**
- `L_crit` 24.78 → 0.40 monotonically — critic converging properly.
- `G_speed` 0.144 → 0.018, `G_comfort` 0.539 → 0.134 — clean monotone improvement.
- α tracks the bottleneck: spread early, concentrates on comfort while comfort is high, redistributes as costs equalise.
- `L_pol` oscillating around 0 in late updates — textbook A2C convergence pattern.

**Concerning:** `G_safety` rising 0.096 → 0.189. Likely benign — initial low value was a stationary policy; policy "waking up" raises the natural floor. Watch whether it plateaus around 0.20 (benign floor) or keeps climbing past 0.25 (real safety regression).

---

## 14. Will the run be transformatively better at PSL?

**A:** Probably modest improvement, not transformative. **Helpful:** 3× more updates, fixed metrics, α showing differentiation between safety/comfort. **Limiting:** architecture is still shallow concatenation of λ; α_speed near zero means EPO can't differentiate speed across λ.

**Predicted eval:** weak diagonal-wins, separation on safety↔comfort but not speed; std bars at N=10 will swamp inter-row differences.

**Test that decides:** section 7.4 TV diagnostic. If TV ≥ 0.15 you have working PSL; if TV < 0.05 you have collapse.

---

## 15. Explaining G, α, L_pol

| Quantity | What it measures | Lives in |
|---|---|---|
| `G` | Environment outcomes — what the policy is achieving | The world |
| `α` | EPO's chosen gradient mix this step | The optimiser |
| `L_pol` | The actual scalar losses being differentiated | The graph |

**L_pol meaning:**
- Strongly negative: entropy bonus dominates (early training).
- Strongly positive: advantages mostly positive (critic lagging a worsening policy).
- **Oscillating near 0: converged-A2C signature.** Critic caught up; policy fine-tuning.

---

## 16. Why is L_pol 3D but L_crit 1D?

**A:** Policy faces a Pareto trade-off; critic doesn't.

- **Policy** must produce one action that affects all 3 costs. ∇L_safety, ∇L_speed, ∇L_comfort point in different (often conflicting) directions. EPO needs them separately to compute α. → 3-vector.
- **Critic** is just a regressor predicting `V(s, λ) ∈ ℝ³`. There's no Pareto trade-off in fitting V to G — the gradients add. → scalar loss.

---

## 17. TV = 0.009 — collapsed policy

```
Mean pairwise TV distance: 0.009
→ Policy has collapsed.
```

**A:** Four suggestions ranked by impact:

1. **More updates (least likely to help):** 0.009 means there's no λ-driven signal making it through. Compounding zero gradients over more steps stays near zero.
2. **Higher `p_corner`:** more corner pressure but still capped by architecture.
3. **Deeper λ injection (large effect, modest code change):** re-concatenate λ at every layer or use FiLM modulation. Estimate: TV → 0.10–0.20.
4. **True hypernetwork:** strongest, but unstable in RL and a bigger refactor.

---

## 18. Implementing deep λ injection

**Q:** Take the simplest approach.

**A:** Re-concatenated λ at every layer in [models/policy.py](models/policy.py) and [models/critic.py](models/critic.py):

```python
self.l1 = nn.Linear(obs_dim    + lam_dim, hidden_dim)
self.l2 = nn.Linear(hidden_dim + lam_dim, hidden_dim)
self.l3 = nn.Linear(hidden_dim + lam_dim, n_actions)

def forward(self, obs, lam):
    h = torch.relu(self.l1(torch.cat([obs, lam], dim=-1)))
    h = torch.relu(self.l2(torch.cat([h,   lam], dim=-1)))
    return self.l3(torch.cat([h, lam], dim=-1))
```

Param count went 74,501 → 75,284 (+1%). Same forward signature; rest of training pipeline unchanged.

---

## 19. Deep RL overfitting?

**Q:** Is there overfitting? Would reducing parameters help?

**A:** **No, the problem is *underfitting* of the λ-conditioning, not overfitting.**

- Training-time `G` ≈ eval-time `G` — no generalisation gap.
- TV = 0.009 means the network failed to fit the λ-dependence task, not memorised it.
- 75K params is on the small end. Reducing further would just lose representation capacity for both driving and λ-differentiation.

What helps: deep λ injection, higher `p_corner`, entropy annealing, more episodes per λ.

What would *demonstrate* overfitting: train on Dirichlet(1) interior + corners, evaluate on extreme corners — if cost is much worse, that's λ-distribution overfitting.

---

## 20. Faster training

**Q:** Need faster training.

**A:** Three levers applied:

| Change | Was → Now | Effect |
|---|---|---|
| `env.duration` | 40 → 25 s | ~38% fewer env steps |
| `policy.hidden_dim` | 256 → 128 | ~30% faster forward/backward |
| `n_episodes_per_pref` | 10 → 6 | 40% fewer rollouts per update |
| `n_updates` | 150 → 100 | 33% fewer total updates |
| `p_corner` | 0.25 → 0.4 | (not speedup — pair with deep λ injection) |

Expected ~3× faster wall-clock end-to-end.

---

## 21. Pygame deprecation warning spam

**Q:** Why am I seeing dozens of pygame `pkg_resources` deprecation warnings?

**A:** Multiprocessing. With `n_workers=16`, each worker is a fresh Python process and each one imports `highway-env` → `pygame` independently. The notebook's `warnings.filterwarnings("ignore")` only applies to the parent kernel; spawned workers don't inherit it.

**Fix:** add `warnings.filterwarnings("ignore")` at the top of [training/parallel_worker.py](training/parallel_worker.py) so each spawned worker silences them.

---

## 22. Bug in objectives.py — iTTC dead zone

**Q:** Can changes be made in [envs/objectives.py](envs/objectives.py) to address low λ-sensitivity?

**A:** Yes — significant bug.

### Bug 1 (significant): iTTC dead zone

```python
candidate_ittc = float(np.clip((TTC_THRESHOLD - ttc) / TTC_THRESHOLD, 0.0, 1.0))
```

| TTC | Old iTTC |
|---|---|
| 1 s | 0.8 |
| **5 s** | **0** ← clip kicks in |
| 10 s | 0 (clipped) |
| 20 s | 0 (clipped) |
| ∞ | 0 |

**Above 5 s, `iTTC = 0`** — exactly the safe-driving regime. So during normal highway following (TTC = 8–20 s), `f_safety` only comes from `dist_violation` (which only fires below MIN_SAFE_DIST = 10 m). For the vast majority of timesteps, **`∂f_safety/∂action = 0`**. The policy gets no gradient pressure to be "even safer."

This directly hurts λ-conditioning: even a perfectly conditioning architecture can't differentiate safety-only from comfort-only λ if both produce zero safety gradient most of the time.

### Fix

CLAUDE.md specifies the smooth formula: `iTTC = clip(TTC_THRESHOLD / TTC, 0, 1)`.

| TTC | New iTTC |
|---|---|
| 1 s | 1 (capped) |
| 5 s | 1 |
| 10 s | 0.5 |
| 20 s | 0.25 |
| ∞ | 0 |

Smooth gradient at all reasonable TTC values. Applied as one-line change in [envs/objectives.py](envs/objectives.py).

### Bug 2 (medium): lateral acceleration formula

[envs/objectives.py:127](envs/objectives.py#L127) uses `(|steering|/WHEELBASE) * v²`. CLAUDE.md says this is wrong for DiscreteMetaAction; should track `_prev_lateral_velocity` and use `|Δvy/dt|`. Less critical; not fixed in this session.

### Bug 3 (cosmetic): weights drifted from CLAUDE.md spec

Code has `W_ITTC=0.6, W_DIST=0.4`. Spec says `0.7, 0.3`.

---

## 23. PSL is important — λ-scaling at input

**Q:** Make changes to make λ matter more at evaluation time. PSL is important.

**A:** Single highest-leverage change: scale λ at input by a fixed constant.

In [models/policy.py](models/policy.py) and [models/critic.py](models/critic.py):
```python
LAM_SCALE = 10.0
...
def forward(self, obs, lam):
    lam_s = lam * LAM_SCALE
    h = torch.relu(self.l1(torch.cat([obs, lam_s], dim=-1)))
    ...
```

**Why this targets collapse:** The previous network could collapse by sending λ-related weights toward zero (small to begin with: 3 channels of 28). Now λ enters at magnitude 10, dominating unit-scale obs values. To ignore λ, the network would have to *actively train* large weights to zero.

**Verification at random init:** L2 difference between logits at corner λ values went from 0.12 → 1.4–1.5. Order-of-magnitude stronger structural λ-dependence.

---

## 24. Why does diagnostic disagree with eval?

**Q:** Diagnostic shows λ sensitivity working; eval rows look nearly identical and noisy.

```
safety-only  → safety=0.400  speed=0.171  comfort=0.088  (±[0.157 0.006 0.098])
speed-only   → safety=0.423  ...
...
```

**A:** Two genuinely different things being measured:

- **TV diagnostic (7.4):** local difference in `π(·|s, λ)` at fixed observations.
- **Eval (8):** episode outcomes under **greedy** action selection.

They disagree because:

1. **Greedy is insensitive to small prob shifts.** A distribution shift from `[0.4, 0.3, ...]` to `[0.3, 0.4, ...]` registers TV = 0.10 but `argmax` doesn't change.
2. **Std swamps the diagonal-wins effect.** Spread between rows ≈ 0.04 vs std ≈ 0.15, SE ≈ 0.024 → diagonal wins are barely 1.5–2 SE above noise.
3. **State distributions differ across λ** — eval conflates "policy response to λ" with "trajectory drift due to early policy differences."
4. **Speed objective saturates** at 0.17 — the env physically caps speed in heavy traffic.

---

## 25. Better evaluation metrics than mean per-step cost

**Q:** Is mean per-step cost the best metric to show in PSL eval?

**A:** **No.** Weaknesses:

1. Hides tail events (rare close-calls, hard brakes) where policy differences live.
2. Speed is saturated.
3. Conflates behaviour with episode duration (crashes truncate denominator).
4. Mean is the wrong reduction for safety-critical metrics.

### Better metrics

- **Action-frequency fingerprints** per λ. Direct behavioural evidence; doesn't depend on env compression.
- **Tail safety:** `mean(min-TTC)`, `P5(TTC)`, **crash-rate** (binary, low variance).
- **Episode-level cumulative cost** (the integral, not the mean) for the Pareto-front plot.

---

## 26. Adding action-frequency cell

**Q:** Add an action-frequency-per-λ cell to the notebook.

**A:** Inserted section 8.1 in [explore.ipynb](explore.ipynb):

- Markdown header explaining the metric.
- Code cell that runs 20 greedy episodes per corner λ, counts action selections, prints a frequency table, plots grouped bars, and reports pairwise TV between action-frequency distributions.

**What working PSL would show:** safety-only leans on SLOWER, speed-only leans on FASTER, comfort-only leans on IDLE. Pairwise TV ≈ 0.4–0.6.

**What collapse looks like:** all rows have nearly identical histograms. Pairwise TV ≈ 0.0–0.05.

---

## 27. Increasing traffic in the environment

**Q:** Is there a way to increase traffic so safety/comfort are tested more per timestep?

**A:** Three levers in env config, applied to [configs/default.yaml](configs/default.yaml):

```yaml
env:
  lanes_count: 3            # was 4 — same cars on fewer lanes
  vehicles_count: 50        # was 30 — more lead-vehicle interactions
  vehicles_density: 1.5     # was 1.0 (default) — tighter spawn
  duration: 25
  policy_frequency: 2
  simulation_frequency: 15
```

Smoke test confirmed dramatically richer per-step signal:

| Metric | Before | After |
|---|---|---|
| Mean safety cost (random actions) | ~0.10 | **0.618** |
| Mean comfort cost | ~0.30 | **0.507** |
| Visible neighbours per step | 2–3 of 4 | **4 of 4** |

### Caveats

- Higher starting costs in PSL log (signal becoming honest).
- More crashes early.
- Speed will saturate further (~0.20–0.25 across all λ); don't expect speed differentiation.
- ~67% slower env step but at `n_workers=16` not a major issue.

---

## Summary of code changes made during this session

| File | Change |
|---|---|
| [models/policy.py](models/policy.py) | Deep λ injection at every layer; `LAM_SCALE = 10.0` multiplier |
| [models/critic.py](models/critic.py) | Same deep injection + scaling, imported from policy |
| [envs/objectives.py](envs/objectives.py) | iTTC formula fix: `clip(TTC_THRESHOLD / TTC, 0, 1)` |
| [training/psl_trainer.py](training/psl_trainer.py) | `sample_preferences` + `p_corner` knob; `PSLHistory` adds per-objective `L_policy`; correct `L_critic` aggregation |
| [training/parallel_worker.py](training/parallel_worker.py) | Returns scalar policy/critic loss + per-objective policy loss vector; `warnings.filterwarnings("ignore")` for worker subprocesses |
| [configs/default.yaml](configs/default.yaml) | `hidden_dim` 256→128; `duration` 40→25; `lanes_count` 4→3; `vehicles_count` 30→50; `vehicles_density: 1.5` |
| [explore.ipynb](explore.ipynb) | Updated `collect_episode` unpacking with entropies; mini-A2C entropy bonus; section 7.3 corrected metrics + L_policy panel; section 7.4 TV diagnostic; section 8.1 action-frequency fingerprints; PSL config tuned multiple times |

---

## Open questions / next steps

1. Re-train with current config (deep λ injection + λ-scaling + corner sampling + iTTC fix + denser traffic) and re-check section 7.4 TV and section 8.1 action-frequencies.
2. If TV ≥ 0.15 and action-frequency rows differ, PSL is working — proceed to write up the Pareto front results.
3. If TV < 0.05, fall back to training K specialists at fixed corner λ via [baselines/scalarized_trainer.py](baselines/scalarized_trainer.py) and present the Pareto front from those discrete points instead.
4. Lateral-acceleration formula in [envs/objectives.py](envs/objectives.py) still diverges from CLAUDE.md spec — fix if comfort needs more sensitivity.
