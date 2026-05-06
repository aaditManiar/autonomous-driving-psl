"""
Helper functions for Experiment 4 — PSL vs Linear Scalarization sweeps.

Provides training, evaluation, convergence diagnostics, and visualization.
All functions accept explicit parameters rather than relying on notebook globals.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────

ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
OBJ_NAMES = ["f_safety", "f_speed", "f_comfort"]
PREF_COLORS = {
    "safety": "#d62728",
    "speed": "#1f77b4",
    "comfort": "#2ca02c",
    "uniform": "#9467bd",
}


def _top_actions(freqs: list | np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k action names and frequencies from a 5-action frequency vector."""
    arr = np.array(freqs, dtype=float)
    if arr.size == 0:
        return []
    order = np.argsort(arr)[::-1][:k]
    return [(ACTION_NAMES[int(i)], float(arr[i])) for i in order if arr[i] > 0]


def _fmt_top_actions(freqs: list | np.ndarray, k: int = 3) -> str:
    tops = _top_actions(freqs, k=k)
    if not tops:
        return "-"
    return ", ".join(f"{name}:{pct * 100:.0f}%" for name, pct in tops)


def _fmt_cost_with_std(mean: float, std: float | None) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


# ── Checkpoint path helpers ──────────────────────────────────────────────────


def psl_ckpt_dir(sweep_label: str, condition: str) -> str:
    return f"checkpoints/exp4/{sweep_label}/psl_{condition}"


def baseline_ckpt(sweep_label: str, condition: str, preset: str, n_eps: int) -> str:
    return (
        f"checkpoints/exp4/{sweep_label}/baseline_{condition}/{preset}/ep{n_eps:05d}.pt"
    )


def _history_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "training_history.json")


# ── Convergence check ────────────────────────────────────────────────────────


def _is_converged(
    history,
    window: int = 60,
    threshold: float = 0.08,
) -> bool:
    """True when all 4 training metrics are flat in the last `window` updates.

    Accepts either a PSLHistory dataclass or the dict loaded from JSON.
    Criterion: (max - min) / mean < threshold for G_safety, G_speed, G_comfort, L_critic.
    """
    get = (
        (lambda k: getattr(history, k, []))
        if hasattr(history, "G_safety")
        else (lambda k: history.get(k, []))
    )
    for key in ["G_safety", "G_speed", "G_comfort", "L_critic"]:
        vals = get(key)
        if len(vals) < window:
            return False
        tail = np.array(vals[-window:], dtype=float)
        if (tail.max() - tail.min()) / (abs(float(np.mean(tail))) + 1e-8) >= threshold:
            return False
    return True


# ── Loading ──────────────────────────────────────────────────────────────────


def load_latest_psl(ckpt_dir: str, env_cfg: dict, psl_cfg, pc: dict):
    """Load the most recent PSL checkpoint into an eval-only trainer (no worker pool)."""
    from training.psl_trainer import PSLTrainer
    from dataclasses import replace as dc_replace

    trainer = PSLTrainer(
        dc_replace(psl_cfg, n_workers=1), env_config=env_cfg, policy_config=pc
    )
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "psl_upd*.pt")))
    if ckpts:
        trainer.load(ckpts[-1])
        print(f"    loaded {os.path.basename(ckpts[-1])}")
    else:
        print(f"    WARNING: no checkpoint found in {ckpt_dir}")
    return trainer


def load_baseline_policy(ckpt_path: str, pc: dict):
    """Load a saved scalarized baseline policy. Returns None if not found."""
    from models.policy import ConditionedPolicy

    if not os.path.exists(ckpt_path):
        print(f"    WARNING: {ckpt_path} not found")
        return None
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = ConditionedPolicy(**pc)
    pol.load_state_dict(saved["policy_state_dict"])
    pol.eval()
    return pol


def load_histories(sweep_label: str, conditions: list) -> dict:
    """Load saved training histories for convergence plotting."""
    histories = {}
    for cond in conditions:
        path = _history_path(psl_ckpt_dir(sweep_label, cond))
        if os.path.exists(path):
            with open(path) as f:
                histories[cond] = json.load(f)
        else:
            print(f"    No history found for {sweep_label}/{cond}")
    return histories


# ── PSL training ─────────────────────────────────────────────────────────────


def run_psl_sweep(
    sweep_label: str,
    sweep_cfgs: dict,
    psl_cfg,
    pc: dict,
    skip: bool = False,
    conv_window: int = 60,
    conv_threshold: float = 0.08,
    conv_min_steps: int = 100,
    check_every: int = 20,
) -> tuple[dict, dict]:
    """Train (or reload from checkpoint) PSL for every condition in the sweep.

    Auto-resume: if psl_upd*.pt already exists in the ckpt_dir (e.g. from a
    previous run), it is loaded automatically and training continues from there.
    This lets you resume the v2 run (150 steps) for the remaining updates.

    Early-stop: checks convergence every `check_every` updates once
    `conv_min_steps` new updates have been added. Stops when all four metrics
    (G_safety, G_speed, G_comfort, L_critic) are flat for `conv_window`
    consecutive updates (range < conv_threshold × mean).

    History merging: new training data is appended to any existing
    training_history.json so the convergence plot always shows the full
    trajectory from step 0.

    Returns
    -------
    trainers  : {condition: PSLTrainer}  — eval-only (n_workers=1)
    histories : {condition: dict}        — full training history (old + new)
    """
    from training.psl_trainer import PSLTrainer
    from dataclasses import replace as dc_replace

    trainers = {}
    histories = {}

    for condition, env_cfg in sweep_cfgs.items():
        ckpt_dir = psl_ckpt_dir(sweep_label, condition)
        os.makedirs(ckpt_dir, exist_ok=True)

        if not skip:
            print(f'\n{"=" * 55}')
            l, v = env_cfg["lanes_count"], env_cfg["vehicles_count"]
            print(f"  Training PSL  sweep={sweep_label}  cond={condition}  L={l} V={v}")
            print(f'{"=" * 55}')

            trainer = PSLTrainer(
                dc_replace(psl_cfg, checkpoint_dir=ckpt_dir),
                env_config=env_cfg,
                policy_config=pc,
            )

            # Auto-resume from the latest existing checkpoint
            existing = sorted(glob.glob(os.path.join(ckpt_dir, "psl_upd*.pt")))
            if existing:
                trainer.load(existing[-1])
                print(
                    f"    Resuming from {os.path.basename(existing[-1])}  "
                    f"(global step {trainer._step})"
                )
            else:
                print("    Starting from scratch")

            # Load old history to append to (full trajectory in convergence plots)
            hpath = _history_path(ckpt_dir)
            old_hist: dict = {}
            if os.path.exists(hpath):
                with open(hpath) as f:
                    old_hist = json.load(f)

            # Manual update loop — runs n_updates ADDITIONAL steps from current step
            converged_early = False
            for i in range(psl_cfg.n_updates):
                info = trainer.update_once()
                global_step = trainer._step

                if (i + 1) % psl_cfg.log_interval == 0:
                    G, a = info["G"], info["alpha"]
                    print(
                        f"PSL upd {global_step:4d}  "
                        f"G=[s={G[0]:.3f} v={G[1]:.3f} c={G[2]:.3f}]  "
                        f"α=[{a[0]:.2f} {a[1]:.2f} {a[2]:.2f}]  "
                        f"L_crit={info['L_critic']:.4f}  "
                        f"crash={info['crash_rate']:.1%}"
                    )

                if (i + 1) % psl_cfg.save_interval == 0:
                    trainer.save(os.path.join(ckpt_dir, f"psl_upd{global_step:04d}.pt"))

                # Convergence check — only once we have enough new history
                if (
                    (i + 1) % check_every == 0
                    and len(trainer.history.G_safety) >= conv_min_steps
                    and _is_converged(trainer.history, conv_window, conv_threshold)
                ):
                    print(
                        f"  *** Converged at global step {global_step} — stopping early ***"
                    )
                    trainer.save(os.path.join(ckpt_dir, f"psl_upd{global_step:04d}.pt"))
                    converged_early = True
                    break

            if not converged_early:
                trainer.save(os.path.join(ckpt_dir, f"psl_upd{trainer._step:04d}.pt"))

            # Serialise new history (alpha_mean contains np.ndarray)
            new_hist = dataclasses.asdict(trainer.history)
            new_hist["alpha_mean"] = [
                a.tolist() if hasattr(a, "tolist") else list(a)
                for a in new_hist["alpha_mean"]
            ]
            # Merge: append new updates to old history
            merged = {k: old_hist.get(k, []) + new_hist.get(k, []) for k in new_hist}
            with open(hpath, "w") as f:
                json.dump(merged, f)

            trainer.close()

        print(f"  Loading PSL {sweep_label}/{condition}")
        trainers[condition] = load_latest_psl(ckpt_dir, env_cfg, psl_cfg, pc)
        hpath = _history_path(ckpt_dir)
        if os.path.exists(hpath):
            with open(hpath) as f:
                histories[condition] = json.load(f)

    return trainers, histories


# ── Baseline training ────────────────────────────────────────────────────────


def _train_one_baseline(args) -> None:
    """Pickleable worker for parallel baseline training. Must be a top-level function."""
    sweep_label, condition, env_cfg, preset, n_eps = args
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import yaml
    from baselines.scalarized_trainer import train as _train

    cfg_path = os.path.join(repo_root, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["env"].update(env_cfg)
    ckpt_dir = os.path.join(
        repo_root, f"checkpoints/exp4/{sweep_label}/baseline_{condition}"
    )
    _train(
        lam_name=preset,
        cfg=cfg,
        checkpoint_dir=ckpt_dir,
        n_episodes_override=n_eps,
        log_interval_override=100,
    )


def run_baseline_sweep(
    sweep_label: str,
    sweep_cfgs: dict,
    presets: list,
    n_episodes: int,
    pc: dict,
    skip: bool = False,
) -> dict:
    """Train (or reload) all baseline presets per condition; presets run in parallel.

    Returns {(condition, preset): ConditionedPolicy}.
    """
    policies = {}
    for condition, env_cfg in sweep_cfgs.items():
        to_train = [
            p
            for p in presets
            if not (
                skip
                and os.path.exists(baseline_ckpt(sweep_label, condition, p, n_episodes))
            )
        ]

        if to_train:
            l, v = env_cfg["lanes_count"], env_cfg["vehicles_count"]
            print(
                f"\n  Baselines {sweep_label}/{condition} (L={l} V={v}): "
                f"training {to_train} in parallel ..."
            )
            args_list = [
                (sweep_label, condition, env_cfg, p, n_episodes) for p in to_train
            ]
            with ProcessPoolExecutor(max_workers=len(to_train)) as ex:
                futs = {ex.submit(_train_one_baseline, a): a[3] for a in args_list}
                for fut in as_completed(futs):
                    p_done = futs[fut]
                    try:
                        fut.result()
                        print(f"    done: {p_done}")
                    except Exception as e:
                        print(f"    ERROR {p_done}: {e}")
        else:
            print(
                f"  Baselines {sweep_label}/{condition}: all checkpoints found, skipping."
            )

        for preset in presets:
            pol = load_baseline_policy(
                baseline_ckpt(sweep_label, condition, preset, n_episodes), pc
            )
            if pol is not None:
                policies[(condition, preset)] = pol

    return policies


# ── Evaluation (parallel) ────────────────────────────────────────────────────


def _eval_episode_worker(args):
    """Run one evaluation episode. Top-level function required for multiprocessing."""
    import warnings as _w

    _w.filterwarnings("ignore")
    policy_sd_np, pc_cfg, lam_np, env_cfg, seed, greedy = args
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import numpy as _np
    import torch as _t
    from models.policy import ConditionedPolicy
    from envs import MOHighwayEnv

    if seed is not None:
        _np.random.seed(int(seed) % (2**32))
        _t.manual_seed(int(seed) % (2**32))

    policy = ConditionedPolicy(**pc_cfg)
    policy.load_state_dict(
        {k: _t.from_numpy(v.copy()) for k, v in policy_sd_np.items()}
    )
    policy.eval()

    lam_t = _t.tensor(lam_np, dtype=_t.float32)
    counts = _np.zeros(5)
    total = 0
    step_costs = []

    env = MOHighwayEnv(config=env_cfg)
    obs, _ = env.reset(seed=int(seed) if seed is not None else None)
    done = False
    with _t.no_grad():
        while not done:
            obs_t = _t.tensor(obs.flatten(), dtype=_t.float32)
            logits = policy(obs_t, lam_t)
            action = (
                int(logits.argmax().item())
                if greedy
                else int(_t.distributions.Categorical(logits=logits).sample())
            )
            obs, cost_vec, term, trunc, _ = env.step(action)
            counts[action] += 1
            total += 1
            step_costs.append(_np.array(cost_vec, dtype=_np.float32))
            done = term or trunc
    env.close()

    return counts.tolist(), total, _np.mean(step_costs, axis=0).tolist()


def eval_policy(
    policy,
    env_cfg: dict,
    eval_lams: dict,
    n_episodes: int = 15,
    greedy: bool = True,
    pc: dict | None = None,
    n_eval_workers: int = 8,
) -> dict:
    """Evaluate at each lambda in eval_lams.

    When `pc` is provided, all n_episodes for each lambda run in parallel via
    ProcessPoolExecutor (one process per episode). This reduces evaluation time
    from O(n_episodes × episode_length) to O(episode_length).

    Falls back to sequential if pc is None.

    Returns {lam_label: {action_freqs (5,), top_actions, costs (3,), cost_stds (3,)}}.
    """
    parallel = (pc is not None) and (n_eval_workers > 1)
    policy_sd_np = (
        {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
        if parallel
        else None
    )
    rng = np.random.default_rng(42)
    eval_seeds = rng.integers(0, 2**31, size=n_episodes)

    out = {}
    for lam_label, lam in eval_lams.items():
        lam_np = np.array(lam, dtype=np.float32)

        if parallel:
            args_list = [
                (policy_sd_np, pc, lam_np, env_cfg, int(s), greedy)
                for s in eval_seeds
            ]
            workers = min(n_eval_workers, n_episodes)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                episode_results = list(ex.map(_eval_episode_worker, args_list))
        else:
            from envs import MOHighwayEnv

            lam_t = torch.tensor(lam_np, dtype=torch.float32)
            episode_results = []
            env = MOHighwayEnv(config=env_cfg)
            with torch.no_grad():
                for seed in eval_seeds:
                    np.random.seed(int(seed) % (2**32))
                    torch.manual_seed(int(seed) % (2**32))
                    obs, _ = env.reset(seed=int(seed))
                    done, step_costs, counts_ep, total_ep = False, [], np.zeros(5), 0
                    while not done:
                        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32)
                        logits = policy(obs_t, lam_t)
                        action = (
                            int(logits.argmax().item())
                            if greedy
                            else int(
                                torch.distributions.Categorical(logits=logits).sample()
                            )
                        )
                        obs, cost_vec, term, trunc, _ = env.step(action)
                        counts_ep[action] += 1
                        total_ep += 1
                        step_costs.append(np.array(cost_vec, dtype=np.float32))
                        done = term or trunc
                    episode_results.append(
                        (
                            counts_ep.tolist(),
                            total_ep,
                            np.mean(step_costs, axis=0).tolist(),
                        )
                    )
            env.close()

        total_counts = np.zeros(5)
        total_steps = 0
        all_costs = []
        for counts_ep, steps_ep, costs_ep in episode_results:
            total_counts += counts_ep
            total_steps += steps_ep
            all_costs.append(costs_ep)

        freqs = total_counts / max(total_steps, 1)
        costs_arr = np.array(all_costs, dtype=float)
        out[lam_label] = {
            "dominant_action": ACTION_NAMES[int(np.argmax(freqs))],
            "action_freqs": freqs.tolist(),
            "top_actions": [
                {"action": name, "freq": freq}
                for name, freq in _top_actions(freqs, k=3)
            ],
            "costs": costs_arr.mean(axis=0).tolist(),
            "cost_stds": costs_arr.std(axis=0).tolist(),
            "n_eval_episodes": int(len(costs_arr)),
        }
    return out


def _ensure(results: dict, sweep_label: str, condition: str) -> None:
    results.setdefault(sweep_label, {}).setdefault(
        condition, {"psl": {}, "baseline": {}}
    )


def eval_sweep(
    sweep_label: str,
    psl_trainers: dict,
    baseline_policies: dict,
    sweep_cfgs: dict,
    psl_eval_lams: dict,
    baseline_eval_lams: dict,
    n_eval: int,
    results: dict,
    pc: dict | None = None,
    n_eval_workers: int = 8,
) -> None:
    """Evaluate all policies and populate results in-place.

    Pass `pc` to enable parallel episode collection (one process per episode).
    """
    for condition, env_cfg in sweep_cfgs.items():
        print(f"\nEvaluating {sweep_label}/{condition} ...")

        if condition in psl_trainers:
            for lam_label, ed in eval_policy(
                psl_trainers[condition].policy,
                env_cfg,
                psl_eval_lams,
                n_eval,
                pc=pc,
                n_eval_workers=n_eval_workers,
            ).items():
                c = ed["costs"]
                s = ed.get("cost_stds", [None, None, None])
                top = _fmt_top_actions(ed.get("action_freqs", []), k=3)
                _ensure(results, sweep_label, condition)
                results[sweep_label][condition]["psl"][lam_label] = ed
                print(
                    f"  PSL  @{lam_label:8s}: "
                    f"saf={_fmt_cost_with_std(c[0], s[0])} "
                    f"spd={_fmt_cost_with_std(c[1], s[1])} "
                    f"com={_fmt_cost_with_std(c[2], s[2])}  "
                    f"actions: {top}"
                )

        for preset, lam in baseline_eval_lams.items():
            key = (condition, preset)
            if key in baseline_policies:
                ed = eval_policy(
                    baseline_policies[key],
                    env_cfg,
                    {preset: lam},
                    n_eval,
                    pc=pc,
                    n_eval_workers=n_eval_workers,
                )[preset]
                c = ed["costs"]
                s = ed.get("cost_stds", [None, None, None])
                top = _fmt_top_actions(ed.get("action_freqs", []), k=3)
                _ensure(results, sweep_label, condition)
                results[sweep_label][condition]["baseline"][preset] = ed
                print(
                    f"  Base @{preset:8s}: "
                    f"saf={_fmt_cost_with_std(c[0], s[0])} "
                    f"spd={_fmt_cost_with_std(c[1], s[1])} "
                    f"com={_fmt_cost_with_std(c[2], s[2])}  "
                    f"actions: {top}"
                )


# ── Results I/O ──────────────────────────────────────────────────────────────


def save_results(results: dict, path: str = "checkpoints/exp4/results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path: str = "checkpoints/exp4/results.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print("No saved results found.")
    return {}


def print_table(results: dict, sweep_label: str) -> None:
    if sweep_label not in results:
        print(f'No results for "{sweep_label}"')
        return
    hdr = (
        f'  {"Condition":10s} {"Policy":22s} {"Pref":8s} '
        f'{"f_safety":15s} {"f_speed":15s} {"f_comfort":15s} Top actions'
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(f'\n{"=" * len(hdr)}\n  Sweep: {sweep_label}')
    print(hdr)
    print(sep)
    for condition, cdata in results[sweep_label].items():
        for lam_label, ed in cdata.get("psl", {}).items():
            c = ed["costs"]
            s = ed.get("cost_stds", [None, None, None])
            print(
                f'  {condition:10s} {"PSL":22s} {lam_label:8s} '
                f'{_fmt_cost_with_std(c[0], s[0]):15s} '
                f'{_fmt_cost_with_std(c[1], s[1]):15s} '
                f'{_fmt_cost_with_std(c[2], s[2]):15s} '
                f'{_fmt_top_actions(ed.get("action_freqs", []), k=3)}'
            )
        for preset, ed in cdata.get("baseline", {}).items():
            c = ed["costs"]
            s = ed.get("cost_stds", [None, None, None])
            print(
                f'  {condition:10s} {("Baseline-" + preset):22s} {preset:8s} '
                f'{_fmt_cost_with_std(c[0], s[0]):15s} '
                f'{_fmt_cost_with_std(c[1], s[1]):15s} '
                f'{_fmt_cost_with_std(c[2], s[2]):15s} '
                f'{_fmt_top_actions(ed.get("action_freqs", []), k=3)}'
            )
        print(sep)


# ── Convergence diagnostics ──────────────────────────────────────────────────


def _plateau_check(
    values: list, frac: float = 0.30, threshold: float = 0.10
) -> tuple[bool, str]:
    """Return (converged, annotation_string) for the last `frac` of a training trace."""
    n = len(values)
    w = max(10, int(n * frac))
    tail = np.array(values[-w:], dtype=float)
    rng = float(tail.max() - tail.min())
    mean = float(abs(np.mean(tail))) + 1e-8
    ok = (rng / mean) < threshold
    return ok, f'{"✓" if ok else "✗"} {values[-1]:.3f}'


def plot_convergence(
    histories: dict,
    sweep_label: str,
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Plot G_safety / G_speed / G_comfort / L_crit vs update for convergence diagnosis.

    ✓ = flat last 30% (range < 10% of mean)  ✗ = still learning.
    If any metric shows ✗, more training updates are needed before trusting results.
    """
    if not histories:
        print("No histories to plot.")
        return

    conditions = list(histories.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(conditions), 2)))

    metric_cfg = [
        ("G_safety",   "f_safety  (↓)",      (0.0, 0.8)),
        ("G_speed",    "f_speed   (↓)",      (0.0, 0.4)),
        ("G_comfort",  "f_comfort (↓)",      (0.0, 0.65)),
        ("L_critic",   "L_crit    (↓)",      (0, 12)),
        ("crash_rate", "crash rate (↓ → 0)", (0.0, 1.05)),
    ]

    fig, axes_grid = plt.subplots(2, 3, figsize=(18, 10))
    axes = [axes_grid[r][c] for r, c in [(0,0),(0,1),(0,2),(1,0),(1,1)]]
    axes_grid[1][2].set_visible(False)   # hide the unused 6th cell

    for ax, (y_key, ylabel, ylim) in zip(axes, metric_cfg):
        for cond, color in zip(conditions, colors):
            hist = histories[cond]
            if y_key not in hist:
                continue
            ys = hist[y_key]
            xs = hist.get("update", list(range(1, len(ys) + 1)))
            ax.plot(xs, ys, color=color, alpha=0.8, linewidth=1.5, label=cond)

            converged, annot = _plateau_check(ys)
            w = max(10, int(len(ys) * 0.30))
            ax.axvspan(xs[-w], xs[-1], alpha=0.07, color=color)
            ax.annotate(
                annot,
                xy=(xs[-1], ys[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=9,
                color=color,
                va="center",
            )

        ax.set_xlabel("update step", fontsize=11)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Sweep {sweep_label} — convergence diagnostics\n"
        "✓ = flat last 30% (range < 10% of mean)   ✗ = still learning",
        fontsize=14,
        y=1.04,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"convergence_{sweep_label}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Convergence plot → {out_path}")


# ── Cost heatmap ─────────────────────────────────────────────────────────────


def plot_heatmap(
    results: dict,
    sweep_label: str,
    sweep_cfgs: dict,
    psl_prefs: list,
    baseline_presets: list,
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Rows = policies, columns = objectives, shaded by cost.

    Navy rectangles mark the diagonal: PSL@X should be greenest in column f_X.
    """
    conditions = list(sweep_cfgs.keys())
    diag_map = {"safety": 0, "speed": 1, "comfort": 2}

    fig, axes = plt.subplots(len(conditions), 1, figsize=(10, 7 * len(conditions)))
    if len(conditions) == 1:
        axes = [axes]
    im = None

    for ax, condition in zip(axes, conditions):
        cdata = results.get(sweep_label, {}).get(condition, {})
        row_labels, cost_matrix, action_summaries = [], [], []

        for pref in psl_prefs:
            ed = cdata.get("psl", {}).get(pref)
            if ed:
                row_labels.append(f"PSL @ {pref}")
                cost_matrix.append(ed["costs"])
                action_summaries.append(_fmt_top_actions(ed.get("action_freqs", []), k=2))

        row_labels.append("")
        cost_matrix.append([np.nan] * 3)
        action_summaries.append("")

        for preset in baseline_presets:
            ed = cdata.get("baseline", {}).get(preset)
            if ed:
                row_labels.append(f"Base @ {preset}")
                cost_matrix.append(ed["costs"])
                action_summaries.append(_fmt_top_actions(ed.get("action_freqs", []), k=2))

        mat = np.array(cost_matrix, dtype=float)
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.6, aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(OBJ_NAMES, fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=11)

        for ri, (costs, action_summary) in enumerate(zip(cost_matrix, action_summaries)):
            for ci, cost in enumerate(costs):
                if not np.isnan(cost):
                    txt = f"{cost:.3f}"
                    if ci == 1 and action_summary:
                        txt += f"\n{action_summary}"
                    colour = "white" if cost > 0.42 else "black"
                    ax.text(
                        ci,
                        ri,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=colour,
                    )

        for pi, pref in enumerate(psl_prefs):
            if pref not in diag_map:
                continue
            col = diag_map[pref]
            rect = plt.Rectangle(
                (col - 0.5, pi - 0.5),
                1,
                1,
                linewidth=3,
                edgecolor="navy",
                facecolor="none",
            )
            ax.add_patch(rect)

        cfg = sweep_cfgs[condition]
        ax.set_title(
            f'{condition}  (L={cfg["lanes_count"]}, V={cfg["vehicles_count"]})',
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

    if im is not None:
        cbar = plt.colorbar(
            im, ax=axes[-1], label="cost  (lower is better)", shrink=0.55, pad=0.02
        )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("cost  (lower is better)", fontsize=12)
    plt.suptitle(
        f"Sweep {sweep_label} — cost heatmap (green=low=good, red=high=bad)\n"
        "Navy boxes: PSL@X should be greenest in f_X column if λ-conditioning works.",
        fontsize=15,
        y=1.03,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"heatmap_{sweep_label}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Heatmap → {out_path}")


# ── λ-differentiation bar grid ───────────────────────────────────────────────


def plot_lam_grid(
    results: dict,
    sweep_label: str,
    sweep_cfgs: dict,
    psl_prefs: list,
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Grid: row = condition, col = objective. Bar with navy border should be lowest."""
    from matplotlib.patches import Patch

    conditions = list(sweep_cfgs.keys())
    expected_best = {"f_safety": "safety", "f_speed": "speed", "f_comfort": "comfort"}

    fig, axes = plt.subplots(
        len(conditions), 3, figsize=(18, 7 * len(conditions)), sharey=False
    )
    if len(conditions) == 1:
        axes = axes.reshape(1, -1)

    for ri, condition in enumerate(conditions):
        cdata = results.get(sweep_label, {}).get(condition, {})
        for ci, obj_name in enumerate(OBJ_NAMES):
            ax = axes[ri, ci]
            vals = [
                cdata.get("psl", {}).get(p, {}).get("costs", [0, 0, 0])[ci]
                for p in psl_prefs
            ]
            clrs = [PREF_COLORS[p] for p in psl_prefs]
            xs = np.arange(len(psl_prefs))
            ax.bar(xs, vals, color=clrs, alpha=0.85, edgecolor="k", linewidth=0.8)

            best = expected_best.get(obj_name)
            if best and best in psl_prefs:
                bi = psl_prefs.index(best)
                ax.bar(
                    bi,
                    vals[bi],
                    color=clrs[bi],
                    alpha=0.85,
                    edgecolor="navy",
                    linewidth=3.5,
                )

            ax.set_xticks(xs)
            ax.set_xticklabels([p[:4] for p in psl_prefs], fontsize=11)
            ax.set_ylim(0, 0.65)
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=10)
            if ci == 0:
                cfg = sweep_cfgs[condition]
                ax.set_ylabel(
                    f'{condition}\nL={cfg["lanes_count"]} V={cfg["vehicles_count"]}',
                    fontsize=12,
                    fontweight="bold",
                )
            if ri == 0:
                ax.set_title(obj_name, fontsize=13, fontweight="bold")

    legend_handles = [
        Patch(color=PREF_COLORS[p], label=f"PSL @ {p}-λ") for p in psl_prefs
    ]
    fig.legend(
        handles=legend_handles,
        fontsize=11,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
    )
    plt.suptitle(
        f"Sweep {sweep_label} — λ-differentiation grid\n"
        "Navy border = bar that should be lowest for correct λ-conditioning.",
        fontsize=15,
        y=1.01,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"lam_grid_{sweep_label}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"λ-grid → {out_path}")


# ── Cross-sweep summary ──────────────────────────────────────────────────────


def plot_cross_sweep(
    results: dict,
    sweep_a_keys: list,
    sweep_b_keys: list,
    psl_prefs: list,
    baseline_presets: list,
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Three-panel bar chart (one per objective) across all conditions in both sweeps."""
    all_conds = [("density", c) for c in sweep_a_keys] + [
        ("lanes", c) for c in sweep_b_keys
    ]
    labels = [f"{s}\n{c}" for s, c in all_conds]
    x = np.arange(len(all_conds))
    n_prefs = len(psl_prefs)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, obj_idx, obj_name in zip(axes, [0, 1, 2], OBJ_NAMES):
        width = 0.15
        for pi, pref in enumerate(psl_prefs):
            vals = [
                results.get(s, {})
                .get(c, {})
                .get("psl", {})
                .get(pref, {})
                .get("costs", [None] * 3)[obj_idx]
                for s, c in all_conds
            ]
            vals = [v if v is not None else float("nan") for v in vals]
            offset = (pi - n_prefs / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=f"PSL@{pref}",
                color=PREF_COLORS[pref],
                alpha=0.85,
                edgecolor="k",
                linewidth=0.7,
            )

        for preset, ls in zip(["safety", "speed", "comfort"], ["-", "--", ":"]):
            if preset not in baseline_presets:
                continue
            bvals = [
                results.get(s, {})
                .get(c, {})
                .get("baseline", {})
                .get(preset, {})
                .get("costs", [None] * 3)[obj_idx]
                for s, c in all_conds
            ]
            bvals = [v if v is not None else float("nan") for v in bvals]
            ax.plot(
                x,
                bvals,
                "D" + ls,
                color=PREF_COLORS[preset],
                markersize=8,
                linewidth=1.8,
                alpha=0.7,
                label=f"Base-{preset}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(obj_name, fontsize=12)
        ax.set_ylim(0, 0.75)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(obj_name, fontsize=13, fontweight="bold")

    plt.suptitle(
        "Cross-sweep summary: PSL (bars) vs Baselines (dashed diamonds)", fontsize=15
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "cross_sweep.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Cross-sweep plot → {out_path}")


def print_findings(results: dict, sweep_a_cfgs: dict, sweep_b_cfgs: dict) -> None:
    """Auto-generate a key-findings summary from the results dict."""
    print("=" * 65)
    print("EXPERIMENT 4 — KEY FINDINGS")
    print("=" * 65)
    for sweep_label, sweep_cfgs in [("density", sweep_a_cfgs), ("lanes", sweep_b_cfgs)]:
        print(f"\nSweep {sweep_label}:")
        if sweep_label not in results:
            print("  (no results)")
            continue
        for condition in sweep_cfgs:
            cdata = results[sweep_label].get(condition, {})
            psl = cdata.get("psl", {})
            base = cdata.get("baseline", {})
            if not psl:
                print(f"  {condition:10s}: no PSL results")
                continue

            actions = {p: psl[p]["dominant_action"] for p in psl}
            diff_ok = len(set(actions.values())) > 1
            pref_costs = {
                pref: psl[pref]["costs"]
                for pref in ["safety", "speed", "comfort"]
                if pref in psl and "costs" in psl[pref]
            }

            def _best_for(pref: str, obj_idx: int) -> bool | None:
                if pref not in pref_costs or len(pref_costs) < 2:
                    return None
                target = pref_costs[pref][obj_idx]
                return target <= min(v[obj_idx] for v in pref_costs.values()) + 1e-9

            saf_ok = _best_for("safety", 0)
            spd_ok = _best_for("speed", 1)
            com_ok = _best_for("comfort", 2)
            psl_spd = psl.get("speed", {}).get("costs", [None])[0]
            b_spd = base.get("speed", {}).get("costs", [None])[0]
            pareto = (
                b_spd - psl_spd if b_spd is not None and psl_spd is not None else None
            )

            print(
                f'  {condition:10s}: λ-diff={"✓" if diff_ok else "✗"}  '
                f'saf-best={"✓" if saf_ok else "✗" if saf_ok is not None else "?"}  '
                f'spd-best={"✓" if spd_ok else "✗" if spd_ok is not None else "?"}  '
                f'com-best={"✓" if com_ok else "✗" if com_ok is not None else "?"}  '
                f"actions={list(actions.values())}",
                end="",
            )
            if pareto is not None:
                print(f"  Pareto Δsaf(vs speed-base)={pareto:+.3f}", end="")
            print()


# ── 28-point Pareto grid evaluation ─────────────────────────────────────────


def eval_pareto_grid(
    sweep_label: str,
    sweep_cfgs: dict,
    pc: dict,
    ckpt_root: str = "checkpoints/exp4",
    n_per_side: int = 6,
    n_episodes: int = 5,
    n_eval_workers: int = 4,
    greedy: bool = True,
) -> dict:
    """
    Evaluate the PSL policy for every condition in a sweep at a 28-point
    triangular grid on the 3-simplex (n_per_side=6 → 28 points).

    Loads the latest psl_upd*.pt checkpoint from
        ckpt_root/{sweep_label}/psl_{condition}/

    Returns
    -------
    dict: {condition: {'lambdas': np.ndarray (28,3), 'costs': np.ndarray (28,3)}}
    """
    from models.policy import ConditionedPolicy

    # Build grid, clip corners so EPO never divides by zero
    pts = []
    for i in range(n_per_side + 1):
        for j in range(n_per_side + 1 - i):
            pts.append([i, j, n_per_side - i - j])
    lambdas = np.array(pts, dtype=np.float32) / float(n_per_side)
    lambdas = np.clip(lambdas, 1e-3, None)
    lambdas /= lambdas.sum(axis=1, keepdims=True)  # re-normalise after clip

    rng = np.random.default_rng()
    results = {}

    for condition, env_cfg in sweep_cfgs.items():
        ckpt_dir = os.path.join(ckpt_root, sweep_label, f"psl_{condition}")
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "psl_upd*.pt")))
        if not ckpts:
            print(f"  No checkpoints for {sweep_label}/{condition} — skipping")
            continue

        policy = ConditionedPolicy(**pc)
        ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["policy_state"])
        policy.eval()
        policy_sd_np = {
            k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()
        }

        costs = np.zeros((len(lambdas), 3), dtype=np.float32)
        stds = np.zeros((len(lambdas), 3), dtype=np.float32)
        print(
            f"  {sweep_label}/{condition}: {len(lambdas)} λ-points × {n_episodes} episodes ..."
        )

        eval_seeds = rng.integers(0, 2**31, size=n_episodes)
        for idx, lam in enumerate(lambdas):
            args_list = [
                (policy_sd_np, pc, lam, env_cfg, int(s), greedy)
                for s in eval_seeds
            ]
            with ProcessPoolExecutor(max_workers=min(n_eval_workers, n_episodes)) as ex:
                episode_results = list(ex.map(_eval_episode_worker, args_list))
            ep_costs = np.array([ep[2] for ep in episode_results], dtype=np.float32)
            costs[idx] = ep_costs.mean(axis=0)
            stds[idx] = ep_costs.std(axis=0)

        results[condition] = {"lambdas": lambdas, "costs": costs, "stds": stds}
        print(
            f"    done  (cost range: saf=[{costs[:,0].min():.3f},{costs[:,0].max():.3f}]  "
            f"spd=[{costs[:,1].min():.3f},{costs[:,1].max():.3f}]  "
            f"com=[{costs[:,2].min():.3f},{costs[:,2].max():.3f}])"
        )

    return results


def save_pareto_grid(
    grid: dict, sweep_label: str, ckpt_root: str = "checkpoints/exp4"
) -> None:
    path = os.path.join(ckpt_root, f"pareto_grid_{sweep_label}.json")
    serialisable = {
        cond: {
            "lambdas": v["lambdas"].tolist(),
            "costs": v["costs"].tolist(),
            "stds": v.get("stds", np.zeros_like(v["costs"])).tolist(),
        }
        for cond, v in grid.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f)
    print(f"Pareto grid saved → {path}")


def load_pareto_grid(sweep_label: str, ckpt_root: str = "checkpoints/exp4") -> dict:
    path = os.path.join(ckpt_root, f"pareto_grid_{sweep_label}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {
        cond: {
            "lambdas": np.array(v["lambdas"], dtype=np.float32),
            "costs": np.array(v["costs"], dtype=np.float32),
            "stds": np.array(
                v.get("stds", np.zeros_like(v["costs"])), dtype=np.float32
            ),
        }
        for cond, v in raw.items()
    }


# ── Pareto surface visualisation ─────────────────────────────────────────────

_PROJ_COLORS = {(0, 1): "Reds", (0, 2): "Greens", (1, 2): "Blues"}
_PROJ_COLORBY = {(0, 1): 0, (0, 2): 2, (1, 2): 1}  # λ-index used for colour
_BASE_MARK = {"safety": "*", "speed": "D", "comfort": "P", "uniform": "X"}
_BASE_COLOR = {
    "safety": "#e74c3c",
    "speed": "#3498db",
    "comfort": "#2ecc71",
    "uniform": "#9b59b6",
}
_COND_MARKER = ["o", "s", "^", "v", "<", ">"]


def plot_pareto_surface(
    grid_results: dict,
    baseline_results: dict,
    sweep_label: str,
    condition_order: list | None = None,
    save_dir: str = "checkpoints/exp4",
) -> None:
    """
    Three 2D Pareto projections for one sweep: (safety,speed), (safety,comfort),
    (speed,comfort).  PSL's 28-point grid is shown as a scatter coloured by the
    dominant λ component for that projection; the four baselines are overlaid as
    distinct marker symbols.

    One figure with 3 side-by-side subplots is saved to
        save_dir/pareto_surface_{sweep_label}.png
    """
    conditions = condition_order or list(grid_results.keys())

    projections = [
        (0, 1, "f_safety", "f_speed", "colour = λ_safety"),
        (0, 2, "f_safety", "f_comfort", "colour = λ_comfort"),
        (1, 2, "f_speed", "f_comfort", "colour = λ_speed"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        f"Pareto Front — Sweep {sweep_label}  " f"(PSL 28-pt grid  vs  baselines)",
        fontsize=15,
    )

    for ax, (xi, yi, xlabel, ylabel, clabel) in zip(axes, projections):
        cmap_name = _PROJ_COLORS[(xi, yi)]
        color_dim = _PROJ_COLORBY[(xi, yi)]

        # ---- PSL scatter ------------------------------------------------- #
        for ci, cond in enumerate(conditions):
            if cond not in grid_results:
                continue
            lams = grid_results[cond]["lambdas"]  # (28, 3)
            costs = grid_results[cond]["costs"]  # (28, 3)
            sc = ax.scatter(
                costs[:, xi],
                costs[:, yi],
                c=lams[:, color_dim],
                cmap=cmap_name,
                vmin=0.0,
                vmax=1.0,
                marker=_COND_MARKER[ci % len(_COND_MARKER)],
                s=80,
                alpha=0.80,
                edgecolors="grey",
                linewidths=0.5,
                label=cond,
            )

        # ---- Baseline scatter --------------------------------------------- #
        first_cond = conditions[0] if conditions else None
        for preset, bm in _BASE_MARK.items():
            for ci, cond in enumerate(conditions):
                b = baseline_results.get(cond, {}).get(preset)
                if b is None:
                    continue
                c_vec = b.get("costs", [None, None, None])
                if c_vec[xi] is None or c_vec[yi] is None:
                    continue
                ax.scatter(
                    c_vec[xi],
                    c_vec[yi],
                    marker=bm,
                    s=220,
                    color=_BASE_COLOR[preset],
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=6,
                    label=f"Base-{preset}" if cond == first_cond else "_nolegend_",
                )

        # ---- Ideal corner (origin) --------------------------------------- #
        ax.scatter(
            [0],
            [0],
            marker="★" if "★" in dir() else "*",
            s=280,
            color="gold",
            edgecolors="k",
            linewidths=1.2,
            zorder=7,
            label="ideal (0,0)",
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{xlabel} vs {ylabel}\n({clabel})", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 0.82)
        ax.set_ylim(-0.02, 0.82)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.3)

        # Deduplicate legend entries
        handles, leg_labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, leg_labels):
            if l not in seen and not l.startswith("_"):
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=9, loc="upper right", ncol=2)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"pareto_surface_{sweep_label}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Pareto surface → {out}")


# ── LaTeX report generation ───────────────────────────────────────────────────


def generate_sweep_a_latex(
    results: dict,
    tex_path: str = "report/sweep_a_results.tex",
) -> tuple[str, str]:
    """Fill report/sweep_a_results.tex with actual numbers from RESULTS.

    Reads the template at tex_path, replaces every '--' placeholder with the
    corresponding value from results['density'], writes the file in-place, and
    returns (paper_table_str, slides_table_str) for inspection.

    Bold formatting is applied to the best (lowest) value in each of the nine
    objective×condition columns across all eight methods.

    Call after eval_sweep has populated RESULTS['density']:
        generate_sweep_a_latex(RESULTS)
    """
    CONDITIONS = [
        ("empty",   "Empty",   15),
        ("normal",  "Normal",  30),
        ("crowded", "Crowded", 50),
    ]
    PSL_ROWS = [
        ("safety",  r"PSL @ $\lambda_s$", r"PSL $\lambda_s$"),
        ("speed",   r"PSL @ $\lambda_v$", r"PSL $\lambda_v$"),
        ("comfort", r"PSL @ $\lambda_c$", r"PSL $\lambda_c$"),
        ("uniform", r"PSL @ $\lambda_u$", r"PSL $\lambda_u$"),
    ]
    BASE_ROWS = [
        ("safety",  r"Base-$s$", r"Base-$s$"),
        ("speed",   r"Base-$v$", r"Base-$v$"),
        ("comfort", r"Base-$c$", r"Base-$c$"),
        ("uniform", r"Base-$u$", r"Base-$u$"),
    ]

    def _get(cond, group, key):
        try:
            ed = results["density"][cond][group][key]
            return list(ed["costs"]), list(ed.get("cost_stds", [None, None, None]))
        except KeyError:
            return [None, None, None], [None, None, None]

    # Collect all values per (condition_idx, obj_idx) for bolding
    col_vals: dict[tuple, list] = {}
    for ci, (cond, _, _) in enumerate(CONDITIONS):
        for oi in range(3):
            vals = []
            for key, _, _ in PSL_ROWS + BASE_ROWS:
                grp = "psl" if (key, _, _) in [(k, l, s) for k, l, s in PSL_ROWS] else "baseline"
                c, _ = _get(cond, grp, key)
                vals.append(c[oi])
            col_vals[(ci, oi)] = vals

    # Rebuild col_vals correctly (PSL then baseline, separate group lookup)
    for ci, (cond, _, _) in enumerate(CONDITIONS):
        for oi in range(3):
            vals = []
            for key, _, _ in PSL_ROWS:
                c, _ = _get(cond, "psl", key)
                vals.append(c[oi])
            for key, _, _ in BASE_ROWS:
                c, _ = _get(cond, "baseline", key)
                vals.append(c[oi])
            col_vals[(ci, oi)] = vals

    def _fmt(v, ci, oi):
        if v is None:
            return "--"
        valid = [x for x in col_vals[(ci, oi)] if x is not None]
        is_best = valid and abs(v - min(valid)) < 1e-4
        s = f"{v:.3f}"
        return r"\textbf{" + s + "}" if is_best else s

    def _build_rows(rows, group, paper=True):
        label_idx = 1 if paper else 2
        lines = []
        for key, plabel, slabel in rows:
            cells = []
            for ci, (cond, _, _) in enumerate(CONDITIONS):
                c, _ = _get(cond, group, key)
                for oi in range(3):
                    cells.append(_fmt(c[oi], ci, oi))
            label = plabel if paper else slabel
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        return "\n".join(lines)

    # ── Paper table ───────────────────────────────────────────────────────────
    paper_table = r"""\begin{table*}[t]
\centering
\caption{%
  Sweep A --- Traffic density (lanes $= 3$, vehicles $\in \{15, 30, 50\}$).
  Mean per-episode objective costs over $N\!=\!15$ greedy evaluation episodes
  ($\downarrow$~better).
  \textbf{Bold}: best value in each column across all methods.
  PSL variants share one trained network evaluated at different $\lambda$;
  each scalarized baseline is a separate network trained on a fixed $\lambda$.
}
\label{tab:sweep_a}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{l | ccc | ccc | ccc}
\toprule
 & \multicolumn{3}{c|}{\textbf{Empty} ($V = 15$)}
 & \multicolumn{3}{c|}{\textbf{Normal} ($V = 30$)}
 & \multicolumn{3}{c}{\textbf{Crowded} ($V = 50$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Method}
  & $f_s \downarrow$ & $f_v \downarrow$ & $f_c \downarrow$
  & $f_s \downarrow$ & $f_v \downarrow$ & $f_c \downarrow$
  & $f_s \downarrow$ & $f_v \downarrow$ & $f_c \downarrow$ \\
\midrule
""" + _build_rows(PSL_ROWS, "psl", paper=True) + "\n" + r"\midrule" + "\n" \
      + _build_rows(BASE_ROWS, "baseline", paper=True) + r"""
\bottomrule
\end{tabular}
\end{table*}"""

    # ── Slides table ──────────────────────────────────────────────────────────
    slides_table = r"""\begin{tabular}{l | ccc | ccc | ccc}
\toprule
 & \multicolumn{3}{c|}{\textbf{Empty} ($V\!=\!15$)}
 & \multicolumn{3}{c|}{\textbf{Normal} ($V\!=\!30$)}
 & \multicolumn{3}{c}{\textbf{Crowded} ($V\!=\!50$)} \\
 & $f_s$ & $f_v$ & $f_c$
 & $f_s$ & $f_v$ & $f_c$
 & $f_s$ & $f_v$ & $f_c$ \\
\midrule
""" + _build_rows(PSL_ROWS, "psl", paper=False) + "\n" + r"\midrule" + "\n" \
      + _build_rows(BASE_ROWS, "baseline", paper=False) + r"""
\bottomrule
\end{tabular}"""

    # ── Write into the .tex file ──────────────────────────────────────────────
    if os.path.exists(tex_path):
        with open(tex_path) as f:
            src = f.read()

        # Replace the paper table block
        import re
        src = re.sub(
            r"\\begin\{table\*\}.*?\\end\{table\*\}",
            paper_table,
            src,
            count=1,
            flags=re.DOTALL,
        )
        # Replace the slides table block inside the comment section
        src = re.sub(
            r"(%% ---- Slides table.*?%% \\end\{frame\})",
            "%% ---- Slides table (Beamer frame) --------------------------\n"
            "%%\n%% \\begin{frame}{PSL vs.\\ Scalarized Baselines --- Traffic Density}\n"
            "%% \\centering\\footnotesize\n"
            + "\n".join("%% " + l for l in slides_table.splitlines())
            + "\n%% \\vspace{4pt}\n"
            "%% {\\scriptsize $f_s$: safety $\\downarrow$\\quad $f_v$: speed $\\downarrow$"
            "\\quad $f_c$: comfort $\\downarrow$\\quad \\textbf{Bold}: best in column}\n"
            "%% \\end{frame}",
            src,
            count=1,
            flags=re.DOTALL,
        )

        # Replace XX placeholders in narrative (only if values are available)
        with open(tex_path, "w") as f:
            f.write(src)
        print(f"LaTeX written → {tex_path}")
    else:
        print(f"Template not found at {tex_path}; printing tables only.")

    print("\n=== PAPER TABLE ===")
    print(paper_table)
    print("\n=== SLIDES TABLE ===")
    print(slides_table)

    return paper_table, slides_table


# ── Presentation slide plots ──────────────────────────────────────────────────


def load_psl_histories(
    sweep_label: str = "density",
    ckpt_root: str = "checkpoints/exp4",
) -> dict:
    """Load training_history.json files saved by the PSL trainer.

    Returns
    -------
    dict  — {condition: columnar_dict}  where columnar_dict has keys
            'update', 'G_safety', 'G_speed', 'G_comfort',
            'L_critic', 'crash_rate', etc.
    """
    out = {}
    cond_dir = os.path.join(ckpt_root, sweep_label)
    if not os.path.isdir(cond_dir):
        print(f"History directory not found: {cond_dir}")
        return out
    for entry in sorted(os.listdir(cond_dir)):
        hist_path = os.path.join(cond_dir, entry, "training_history.json")
        if not os.path.isfile(hist_path):
            continue
        cond = entry[4:] if entry.startswith("psl_") else entry
        with open(hist_path) as f:
            out[cond] = json.load(f)
    return out


def plot_slides_convergence(
    histories_density: dict,
    save_dir: str = "checkpoints/exp4",
    smooth: int = 10,
) -> None:
    """Plot 1 (mandatory): G_safety + crash rate over training updates.

    Two-panel figure; one line per density condition (empty/normal/crowded).
    A rolling mean with window=smooth is overlaid on the raw trace.

    histories_density : dict  — loaded by load_histories('density') or
      equivalent. Keys are condition labels; values are columnar dicts with
      arrays 'update', 'G_safety', 'crash_rate', etc. (as saved by PSLTrainer).
    """
    COND_STYLE = {
        "empty":   {"color": "#2ca02c", "label": "Empty (V=15)"},
        "normal":  {"color": "#1f77b4", "label": "Normal (V=30)"},
        "crowded": {"color": "#d62728", "label": "Crowded (V=50)"},
    }

    def _smooth(arr, w):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    ax_safe, ax_crash = axes

    for cond, style in COND_STYLE.items():
        hist = histories_density.get(cond)
        if hist is None:
            continue

        # Support both columnar dict (from file) and list-of-row-dicts
        if isinstance(hist, dict):
            upds   = np.array(hist.get("update", list(range(1, len(hist["G_safety"]) + 1))))
            g_safe = np.array(hist["G_safety"],  dtype=float)
            crash  = np.array(hist["crash_rate"], dtype=float) * 100
        else:
            upds   = np.array([h.get("update", i + 1) for i, h in enumerate(hist)])
            g_safe = np.array([h.get("mean_costs", [h.get("G_safety", 0)])[0] for h in hist])
            crash  = np.array([h.get("crash_rate", 0) for h in hist]) * 100

        color = style["color"]
        label = style["label"]

        ax_safe.plot(upds, g_safe, alpha=0.22, color=color, linewidth=0.8)
        ax_safe.plot(upds, _smooth(g_safe, smooth), color=color, linewidth=2.2, label=label)

        ax_crash.plot(upds, crash, alpha=0.22, color=color, linewidth=0.8)
        ax_crash.plot(upds, _smooth(crash, smooth), color=color, linewidth=2.2, label=label)

    ax_safe.set_xlabel("Update", fontsize=13)
    ax_safe.set_ylabel("G_safety (cumulative return, ↓ better)", fontsize=12)
    ax_safe.set_title("Safety Return vs. Training Progress", fontsize=13, fontweight="bold")
    ax_safe.legend(fontsize=11)
    ax_safe.grid(alpha=0.3)

    ax_crash.set_xlabel("Update", fontsize=13)
    ax_crash.set_ylabel("Crash Rate (%)", fontsize=12)
    ax_crash.set_title("Crash Rate vs. Training Progress", fontsize=13, fontweight="bold")
    ax_crash.set_ylim(0, 105)
    ax_crash.legend(fontsize=11)
    ax_crash.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "convergence_density.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Convergence plot → {out}")


def plot_slides_pareto(
    grid_results: dict,
    results: dict,
    sweep_label: str = "density",
    condition: str = "normal",
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Plot 2: f_safety vs f_speed Pareto projection.

    PSL is shown as a continuous curve of 28 Pareto-grid points coloured by
    their λ_safety weight (cool→warm = safety-focused→speed-focused).
    The four scalarized baselines appear as large markers.

    grid_results : dict  — e.g. PARETO_GRID['density']
      keys: condition label → list of dicts with 'lambda' and 'costs'.
    results      : dict  — e.g. RESULTS['density']
      for baseline scatter points.
    """
    PREF_MARKERS = {
        "safety":  ("s", "#d62728", r"Base-$s$"),
        "speed":   ("^", "#1f77b4", r"Base-$v$"),
        "comfort": ("D", "#8c564b", r"Base-$c$"),
        "uniform": ("o", "#7f7f7f", r"Base-$u$"),
    }

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # PSL Pareto curve
    pts = grid_results.get(condition, [])
    if pts:
        lam_s = np.array([p["lambda"][0] for p in pts])
        f_safe = np.array([p["costs"][0] for p in pts])
        f_spd  = np.array([p["costs"][1] for p in pts])

        # Sort by λ_safety so the line traces the front cleanly
        order = np.argsort(lam_s)
        sc = ax.scatter(
            f_safe[order], f_spd[order],
            c=lam_s[order], cmap="RdYlBu_r",
            s=60, zorder=5, label="PSL (Pareto front)",
            vmin=0.05, vmax=0.90,
        )
        ax.plot(f_safe[order], f_spd[order], color="gray", linewidth=1.0,
                alpha=0.5, zorder=4)
        plt.colorbar(sc, ax=ax, label=r"$\lambda_{\mathrm{safety}}$", shrink=0.85)

    # Baseline scatter
    cond_res = results.get(sweep_label, {}).get(condition, {}).get("baseline", {})
    for pref, (marker, color, mlabel) in PREF_MARKERS.items():
        entry = cond_res.get(pref, {})
        costs = entry.get("costs")
        if costs and len(costs) >= 2:
            ax.scatter(
                costs[0], costs[1],
                marker=marker, color=color, s=180,
                edgecolors="k", linewidths=1.2, zorder=6,
                label=mlabel,
            )

    ax.set_xlabel(r"$f_{\mathrm{safety}}$ $\downarrow$", fontsize=13)
    ax.set_ylabel(r"$f_{\mathrm{speed}}$ $\downarrow$", fontsize=13)
    cond_pretty = condition.capitalize()
    ax.set_title(f"Safety–Speed Pareto Front ({cond_pretty})", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(-0.02, 0.82)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"pareto_projection_{condition}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Pareto projection → {out}")


def plot_slides_lambda_bars(
    results: dict,
    sweep_label: str = "density",
    condition: str = "normal",
    save_dir: str = "checkpoints/exp4",
) -> None:
    """Plot 3: λ-conditioning bar chart — PSL@λ vs Base-λ on each objective.

    One subplot per preference preset (safety / speed / comfort), arranged in a
    column (3 rows × 1 col).  Each subplot shows 3 grouped pairs of bars — one
    pair per objective — so the owned objective stands out clearly.
    Solid bars = PSL, hatched bars = scalarized baseline.
    """
    PRESETS = [
        ("safety",  r"$\lambda_s$ (safety)", 0),
        ("speed",   r"$\lambda_v$ (speed)",  1),
        ("comfort", r"$\lambda_c$ (comfort)", 2),
    ]
    OBJ_LABELS  = [r"$f_{\rm safety}$", r"$f_{\rm speed}$", r"$f_{\rm comfort}$"]
    OBJ_COLORS  = ["#d62728", "#1f77b4", "#8c564b"]

    cond_data = results.get(sweep_label, {}).get(condition, {})
    psl_data  = cond_data.get("psl",      {})
    base_data = cond_data.get("baseline", {})

    cond_pretty = condition.capitalize()
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))
    fig.suptitle(
        f"PSL@λ vs Scalarized Baseline — {cond_pretty} Traffic\n"
        r"(solid = PSL, hatched = Baseline; highlighted = owned objective)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    bar_w  = 0.32
    xs     = np.arange(3)          # one position per objective

    for gi, (pref, subplot_title, owned_oi) in enumerate(PRESETS):
        ax = axes[gi]
        psl_costs  = psl_data.get(pref,  {}).get("costs",  [0, 0, 0])
        base_costs = base_data.get(pref, {}).get("costs",  [0, 0, 0])
        if psl_costs  is None: psl_costs  = [0, 0, 0]
        if base_costs is None: base_costs = [0, 0, 0]

        for oi in range(3):
            alpha = 1.0 if oi == owned_oi else 0.45
            color = OBJ_COLORS[oi]
            ax.bar(xs[oi] - bar_w / 2, psl_costs[oi],
                   width=bar_w, color=color, alpha=alpha,
                   edgecolor="k", linewidth=0.8,
                   label="PSL" if oi == 0 else "_")
            ax.bar(xs[oi] + bar_w / 2, base_costs[oi],
                   width=bar_w, color=color, alpha=alpha,
                   hatch="///", edgecolor="k", linewidth=0.8,
                   label="Baseline" if oi == 0 else "_")
            # Value labels
            for x_pos, val in [(xs[oi] - bar_w / 2, psl_costs[oi]),
                                (xs[oi] + bar_w / 2, base_costs[oi])]:
                ax.text(x_pos, val + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=9)

        # Highlight the owned objective column
        ax.axvspan(xs[owned_oi] - 0.45, xs[owned_oi] + 0.45,
                   alpha=0.08, color="gold", zorder=0)

        ax.set_title(subplot_title, fontsize=12, fontweight="bold", loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(OBJ_LABELS, fontsize=12)
        ax.set_ylabel("Cost (↓ better)", fontsize=11)
        ax.set_ylim(0, 0.72)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"lambda_bars_{condition}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Lambda bar chart → {out}")
