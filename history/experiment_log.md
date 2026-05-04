# PSL Experiment Log

Tracks every training run: configuration, results, and what we learned.
Used to compare across experiments and build the final narrative.

Each experiment entry covers: **Changes made**, **Hyperparameters**, **Training metrics**, **Evaluation results**, **Key findings**, **What to try next**.

---

## Experiment 1 — Baseline PSL (before architectural changes)

**Branch:** main (before partner's feature branch was merged)  
**Date:** ~April 2026  
**Status:** Results not saved. Description from memory and chat history.

### Changes relative to project start
None — original implementation as designed.

### Known issues at this stage
- iTTC formula: `clip((TTC_THRESHOLD - ttc) / TTC_THRESHOLD, 0, 1)` — dead zone: iTTC = 0 for any TTC ≥ 5 s (normal driving). Safety gradient was effectively zero during non-critical steps.
- Shallow λ injection: λ concatenated only at the input layer. Downstream layers could ignore it.
- No LAM_SCALE: λ values in [0,1] too small relative to normalized obs (~[-1,1]), easy for weights to zero out.
- No corner sampling: Dirichlet(1) never produces near-corner λ, so corner preferences were never trained.
- L_crit logging was actually `mean(critic_grad²)`, not MSE.

### Results
Not stored. Qualitative description from chat context:
- Policy collapsed toward "always FASTER" within ~75 updates.
- G_safety improved slightly; G_speed/comfort did not differentiate across λ.
- No visible Pareto front.
- Policy completely ignored λ at inference.

### Key learning
Without a strong λ signal, the policy learns one dominant action and ignores the conditioning vector entirely. The dead-zone iTTC meant safety never provided gradient during normal driving.

---

## Experiment 2 — After Architectural Fixes (feature branch)

**Branch:** feature branch (merged to main by Ravi on ~May 2026)  
**Date:** May 2026  
**Config file:** `configs/default.yaml` at time of run  
**Notebook cells:** PSL-14 (config + training), PSL-15 (curves), PSL-16 (TV diagnostic), PSL-18 (adaptation eval), PSL-19 (action fingerprints), PSL-21 (Pareto grid)

### Changes relative to Experiment 1

| Component | Before | After |
|-----------|--------|-------|
| iTTC formula | `clip((TTC_THRESHOLD - ttc) / TTC_THRESHOLD, 0, 1)` — zero for TTC ≥ 5s | `clip(TTC_THRESHOLD / ttc, 0, 1)` — nonzero everywhere, smooth gradient |
| λ injection | Input layer only | Every layer (deep injection) |
| LAM_SCALE | None (λ in [0,1]) | 10.0 (λ scaled to [0,10]) |
| Corner sampling | None (p_corner=0.0) | p_corner=0.4 |
| hidden_dim | 256 | 128 |
| Traffic density | 4 lanes, 30 vehicles | 3 lanes, 50 vehicles, density=1.5 |
| Episode duration | 40 s | 25 s |
| L_crit logging | `mean(grad²)` — wrong | Actual MSE from worker |
| L_policy logging | Not tracked | Per-objective (safety/speed/comfort) |

### Hyperparameters (PSL-14)
```
n_pref_samples      = 16   (K)
n_episodes_per_pref = 6    (N)
n_updates           = 100
gamma               = 0.99
learning_rate       = 3e-4
critic_lr           = 1e-3
entropy_coef        = 0.01
p_corner            = 0.4
grad_clip           = 0.5
n_workers           = 16
seed                = 42
```
Total episodes per run: 16 × 6 × 100 = 9,600

### Training metrics (from PSL-15 log)

| Update | G_safety | G_speed | G_comfort | α (mean) | L_crit |
|--------|----------|---------|-----------|----------|--------|
| 5      | 0.306    | 0.166   | 0.501     | [0.38, 0.50, 0.12] | 22.05 |
| 25     | 0.417    | 0.088   | 0.487     | [0.50, 0.25, 0.25] | 7.33  |
| 50     | 0.432    | 0.058   | 0.498     | [0.50, 0.25, 0.25] | 5.23  |
| 75     | 0.427    | 0.072   | 0.444     | [0.47, 0.06, 0.46] | 4.69  |
| 100    | 0.444    | 0.090   | 0.414     | [0.36, 0.25, 0.39] | 4.47  |

**Interpretation of training trajectory:**
- G_safety worsens (0.306 → 0.444): policy trades safety for other objectives as training proceeds.
- G_speed dramatically improves (0.166 → 0.058): policy learns to go faster on average.
- G_comfort improves (0.501 → 0.414).
- L_crit drops and stabilizes (~4.5): critic converges.
- α values are diverse and dynamic — EPO is not always choosing safety.

### Evaluation results

**λ-conditioning check (PSL-16, stochastic sampling, 8 observations):**
```
safety-only  vs  speed-only   : TV = 0.410  ✓
safety-only  vs  comfort-only : TV = 0.547  ✓
speed-only   vs  comfort-only : TV = 0.357  ✓
Mean pairwise TV              : 0.438
```
→ Action probability distributions ARE differentiated across λ.

**Adaptation evaluation (PSL-18, greedy, 40 episodes per λ):**
```
safety-only  λ=(0.9,0.05,0.05) → safety=0.447  speed=0.175  comfort=0.126
speed-only   λ=(0.05,0.9,0.05) → safety=0.444  speed=0.174  comfort=0.120
comfort-only λ=(0.05,0.05,0.9) → safety=0.466  speed=0.174  comfort=0.117
uniform      λ=(1/3,1/3,1/3)   → safety=0.437  speed=0.175  comfort=0.110
```
→ Outcomes nearly identical across all λ. Policy is not producing differentiated behavior in terms of costs.

**Action frequencies (PSL-19, greedy, 20 episodes per λ):**
```
safety-only  : LANE_LEFT  100%
speed-only   : LANE_LEFT  100%
comfort-only : LANE_RIGHT 100%
uniform      : LANE_LEFT  100%
```
TV between greedy action distributions: 0 (safety vs speed), 1.0 (safety vs comfort), 0 (speed vs uniform).

**Pareto grid (PSL-21, 36 λ points, 5 episodes each, greedy):**
- f_speed: 0.168 – 0.182 (range = **0.014** — essentially frozen)
- f_safety: 0.284 – 0.557 (range = 0.273, no systematic λ-dependence)
- f_comfort: 0.043 – 0.433 (range = 0.390, high noise)

### Key findings

1. **Action distributions are λ-conditioned (TV=0.438) but greedy outcomes are not.**  
   The network has learned to shift its probability mass across λ, but the argmax action is identical for safety/speed/uniform and near-identical for comfort. This gap between stochastic diversity and greedy behavior is the primary failure mode.

2. **f_speed is frozen at ~0.17 across the entire Pareto grid.**  
   The policy never selects FASTER or SLOWER. Lane-change actions (LANE_LEFT/LANE_RIGHT) dominate. Since speed is only controlled by FASTER/SLOWER, there is no speed variation regardless of λ.

3. **EPO α in the synthetic smoke test (PSL-13):**  
   With losses=[0.6, 0.3, 0.1] (safety worst), EPO gives α=[1,0,0] for 3/4 preferences.  
   For preference=[0.1, 0.7, 0.2] (speed-heavy), EPO still focuses on safety.  
   This is expected EPO behavior: it corrects toward the Pareto front before following preferences. When the policy is far from Pareto-optimal (always true in RL), EPO acts like min-max optimization.

4. **The Pareto front appears narrow for this environment.**  
   The physics of highway driving constrain the tradeoffs: going faster inherently reduces TTC and raises safety cost. The achievable range of f_speed (0.17 ± 0.01) is narrow given the environment's traffic density and speed cap.

5. **L_crit converges, confirming the critic is learning.**  
   L_crit stabilizes around 4.5, suggesting the value function estimates are reasonable. The training is not trivially broken.

### Unresolved questions
- Would more training updates (200+) produce clearer λ-separation?
- Is the frozen f_speed caused by policy collapse or a genuine Pareto constraint?
- Does entropy regularization strength (0.01) explain why FASTER is never selected?

---

## Experiment 3 — Jacobian normalisation + higher entropy

**Branch:** main (feature branch merged)
**Date:** May 2026
**Config:** PSL-14 cell

### Changes relative to Experiment 2

| Parameter | Exp 2 | Exp 3 |
|-----------|-------|-------|
| `entropy_coef` | 0.01 | **0.05** |
| `n_episodes_per_pref` (N) | 6 | **10** |
| `n_updates` | 100 | **150** |
| `n_pref_samples` (K) | 16 | **12** |
| `n_workers` | 16 | **11** (matched to M4 Pro cores) |
| Jacobian normalisation before EPO | No | **Yes** — `J_k /= J_k.norm(dim=1, keepdim=True).clamp(1e-8)` |

Total episodes: 12 × 10 × 150 = **18,000**

### Training metrics (final update)

| Update | G_safety | G_speed | G_comfort | Notes |
|--------|----------|---------|-----------|-------|
| 60/150 | 0.499 | 0.118 | 0.386 | Mid-training |
| 150/150 | 0.526 | 0.132 | 0.271 | Final |

G_comfort improved substantially (0.414 → 0.271). G_speed slightly worse (0.090 → 0.132) —
higher entropy allows more random actions including uncomfortable ones, explaining the spread.
G_safety increased further (0.444 → 0.526) — consistent with denser exploration.

### Evaluation results

**Action frequencies (PSL-19, greedy):**
- Dominant action shifted from **LANE_LEFT → IDLE** for safety/speed/uniform λ
- Comfort λ: LANE_RIGHT 100% (unchanged)
- FASTER and SLOWER still never selected

**Adaptation evaluation (PSL-18, greedy):**
- Costs remain nearly identical across all four corner λ values
- No evidence of λ-driven cost differentiation

**Pareto grid (PSL-21):**
- f_speed still approximately frozen (~0.13–0.18 range)
- No systematic λ-correlated structure observed

### Key findings

1. **Jacobian normalisation changed EPO α distribution (positive).**
   α values are more balanced (e.g. [0.44, 0.15, 0.41]) compared to Experiment 2's frequent
   [1, 0, 0] collapses. EPO is now computing more diverse mixing weights. This is progress.

2. **IDLE collapse replaced LANE_LEFT collapse.**
   Higher entropy_coef broke the lane-change loop, but the policy converged to IDLE instead.
   IDLE is physically rational (stable highway following) but still produces no λ-differentiation
   because it is the dominant greedy action for all preferences.

3. **f_speed frozen ≈ 0.13 — FASTER never selected.**
   The safety gradient still trains FASTER away. Under safety-λ, choosing FASTER would increase
   iTTC cost; EPO correctly identifies this but even under speed-λ the policy doesn't select
   FASTER. This suggests the safety gradient dominates during training even with normalised
   Jacobians, because the *value function* has learned that FASTER leads to high returns on
   the safety objective — so the advantage A_safety is strongly negative when FASTER is taken,
   discouraging it even under speed-λ.

4. **PSL-18 does not show expected Pareto pattern.**
   For a correct PSL policy, safety-λ should produce the lowest safety cost across evaluations.
   This is not observed. Root cause: single dominant action (IDLE) means all λ evaluate
   to the same point on the Pareto front.

5. **Safety baseline ≈ 0.5 is structural, not a failure.**
   With iTTC = clip(5/TTC, 0, 1), normal highway following at TTC=10 s scores 0.5.
   Dense traffic (50 vehicles, 3 lanes) means TTC ≈ 10–15 s is the typical state.
   A safety cost of 0.45–0.55 is expected for any reasonable driving policy in this env.

6. **L_crit improved vs Experiment 2** (~1.37 mid-training vs ~4.5 stable in Exp 2).
   Better Jacobian estimates (N=10) and more entropy give the critic cleaner training signal.

### Unresolved questions
- Does the policy need a fundamentally different training signal to learn FASTER under speed-λ?
  Specifically: the advantage for speed may always be negative when FASTER is selected (because
  the safety cost spikes), preventing reinforcement even when λ says "I accept safety cost."
- Should per-objective advantages be computed independently for each objective before mixing,
  or would normalising advantages per-objective help decouple the safety and speed signals?
- Is the IDLE collapse a genuine Pareto minimum (IDLE IS Pareto-optimal for this env) or
  is there a better point the policy is failing to find?

---

---

## Experiment 4 — PSL vs Linear Scalarization: Density & Lane-Geometry Sweeps

**Notebook:** `exp4.ipynb`  
**Date:** May 2026  
**TTC_THRESHOLD:** 3.0 s (patched at import time)  
**n_updates:** 100 (v1, initial run) → 150 (v2, after per-objective advantage normalisation fix)

### Design

Two sweeps, each with 3 conditions:
- **Sweep A (density):** lanes=3 fixed, vehicles ∈ {15 (empty), 30 (normal), 50 (crowded)}
- **Sweep B (lanes):** vehicles=30 fixed, lanes ∈ {2, 3, 4}

PSL config: K=12, N=10, entropy=0.05, p_corner=0.4, n_workers=11  
Baselines: safety / speed / comfort / uniform presets, 600 episodes each  
Evaluation: greedy (argmax), 15 episodes per λ

---

### Sweep A v1 Results (100 updates, no per-objective normalisation)

#### Training trajectory (empty condition, representative)

| Update | G_safety | G_speed | G_comfort | α |
|--------|----------|---------|-----------|---|
| 10 | 0.442 | 0.098 | 0.418 | [0.67, 0.25, 0.08] |
| 50 | 0.437 | 0.088 | 0.474 | [0.55, 0.11, 0.33] |
| 100 | 0.461 | 0.091 | 0.475 | [0.58, 0.05, 0.36] |

α is diverse (not collapsed to [1,0,0]) — EPO + Jacobian normalisation working.  
L_crit drops from 6.3 → 3.3, critic converging.

#### Evaluation results (greedy, 15 episodes)

```
Condition  Policy             Pref     f_safety  f_speed  f_comfort  Dominant
empty      PSL                safety   0.393     0.177    0.124      LANE_LEFT
empty      PSL                speed    0.340     0.174    0.351      LANE_RIGHT
empty      PSL                uniform  0.370     0.173    0.103      LANE_LEFT
empty      Baseline-safety    safety   0.346     0.173    0.069      LANE_RIGHT
empty      Baseline-speed     speed    0.502     0.028    0.142      FASTER
empty      Baseline-uniform   uniform  0.372     0.172    0.069      LANE_LEFT

normal     PSL                safety   0.431     0.180    0.140      LANE_RIGHT
normal     PSL                speed    0.338     0.170    0.098      LANE_LEFT
normal     PSL                uniform  0.428     0.178    0.107      LANE_LEFT
normal     Baseline-safety    safety   0.388     0.175    0.100      LANE_RIGHT
normal     Baseline-speed     speed    0.501     0.024    0.141      FASTER
normal     Baseline-uniform   uniform  0.504     0.024    0.128      FASTER

crowded    PSL                safety   0.346     0.174    0.353      LANE_LEFT
crowded    PSL                speed    0.404     0.174    0.118      LANE_LEFT
crowded    PSL                uniform  0.407     0.178    0.148      LANE_LEFT
crowded    Baseline-safety    safety   0.395     0.174    0.107      LANE_LEFT
crowded    Baseline-speed     speed    0.498     0.024    0.134      FASTER
crowded    Baseline-uniform   uniform  0.361     0.172    0.087      LANE_RIGHT
```

### Key Findings (v1)

**1. Pareto balance confirmed for speed objective (strong result):**  
Baseline-speed achieves f_speed=0.024 by driving FASTER constantly, but pays f_safety=0.50.  
PSL@speed accepts f_speed=0.17 in exchange for f_safety=0.34–0.40. Delta safety = 0.10–0.16.  
This Pareto benefit holds across all three density conditions.

**2. PSL differentiates dominant actions in empty and normal (partial):**  
- empty: safety→LANE_LEFT, speed→LANE_RIGHT (different actions ✓)
- normal: safety→LANE_RIGHT, speed→LANE_LEFT (different actions ✓)
- crowded: ALL three preferences → LANE_LEFT (collapsed ✗)

**3. f_speed shows no λ-differentiation — FASTER never selected by PSL:**  
PSL@safety and PSL@speed both achieve f_speed ≈ 0.17–0.18 in all conditions.  
Speed is controlled entirely via lane positioning, not longitudinal acceleration.  
Only Baseline-speed uses FASTER (recklessly: saf=0.50).

**4. Safety cost ordering inverted in empty/normal (critical failure):**  
Expected: PSL@safety f_safety < PSL@speed f_safety.  
Observed: PSL@speed (0.340/0.338) < PSL@safety (0.393/0.431) in empty and normal.  
In crowded: correctly ordered (0.346 < 0.404).  
Root cause: f_safety costs (~0.4) are 2× larger than f_speed costs (~0.17), giving the  
safety Jacobian row larger norm. Even with EPO α=[0.1, 0.8, 0.1] for speed-λ, safety  
still gets disproportionate gradient contribution. PSL@speed coincidentally learned  
LANE_RIGHT, which is the naturally safer lane in low-density conditions.

**5. Comfort objective missing from v1 evaluation** — baseline and PSL comfort preference not included.

### Fixes Applied for v2 (Sweep A rerun)

| Change | File | Reason |
|--------|------|--------|
| Per-objective advantage normalisation | `training/parallel_worker.py` | Removes cross-objective scale imbalance so α truly determines gradient priority |
| n_updates 100 → 150 | `exp4.ipynb` EXP4-02 | More training for λ-conditioned cost ordering to converge |
| Added 'comfort' to PSL_EVAL_LAMS + BASELINE_PRESETS | `exp4.ipynb` EXP4-02 | All 3 objectives now evaluated |
| Replaced bar charts with heatmap + λ-grid | `exp4.ipynb` EXP4-A4/B4 | Previous charts had 54 bars, unreadable |

**Per-objective normalisation detail:**  
`adv_std = advantages.std(dim=0).clamp(min=1e-8); advantages /= adv_std`  
Normalises across objectives (not across time), so within-objective urgency signal  
(sign, time-ordering) is preserved. Only removes the artificial scale factor from  
cost function design.

### Sweep A v2 Results

**Status:** Pending rerun with fixes above.

---

## Sweep B — Lane Geometry (exp4.ipynb)

**Status:** Not yet run.

---

## Reference: EPO smoke test finding (PSL-13)

EPO behavior with synthetic J ~ N(0,1) and losses=[0.6, 0.3, 0.1]:
```
pref=[0.7, 0.2, 0.1]  →  α=[0., 1., 0.]   (safety pref → EPO focuses speed)
pref=[0.1, 0.7, 0.2]  →  α=[1., 0., 0.]   (speed pref  → EPO focuses safety)
pref=[0.1, 0.2, 0.7]  →  α=[1., 0., 0.]   (comfort pref → EPO focuses safety)
pref=[0.33,0.33,0.33] →  α=[1., 0., 0.]   (uniform     → EPO focuses safety)
```
EPO is not preference-following when far from the Pareto front — this is expected behavior, not a bug. EPO guarantees Pareto-stationarity, not preference-proportional improvement. Linear scalarization (α = λ) would directly follow preferences but makes no Pareto guarantees and misses concave regions of the front.
