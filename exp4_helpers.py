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
        ("G_safety",   "f_safety  (↓)",    (0.0, 0.8)),
        ("G_speed",    "f_speed   (↓)",    (0.0, 0.4)),
        ("G_comfort",  "f_comfort (↓)",    (0.0, 0.65)),
        ("L_critic",   "L_crit    (↓)",    (0, 12)),
        ("crash_rate", "crash rate (↓ → 0)", (0.0, 1.05)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

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

    fig, axes = plt.subplots(1, len(conditions), figsize=(8 * len(conditions), 8))
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
            im, ax=axes[-1], label="cost  (lower is better)", shrink=0.82, pad=0.02
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
        len(conditions), 3, figsize=(18, 5 * len(conditions)), sharey=False
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
