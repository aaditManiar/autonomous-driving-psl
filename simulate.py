"""
simulate.py — Interactive simulation of trained PSL and baseline policies.

Usage examples
--------------
# PSL policy, safety preference, normal traffic, interactive window:
    python simulate.py

# PSL policy, speed preference, crowded traffic:
    python simulate.py --model psl --pref speed --condition crowded

# Scalarized speed baseline, normal traffic:
    python simulate.py --model baseline --pref speed

# Save a video instead of opening a window:
    python simulate.py --model psl --pref comfort --video

# Run all four PSL preferences back-to-back and save videos:
    python simulate.py --model psl --pref all --video

# Sweep B (lanes): 2-lane road, PSL safety preference:
    python simulate.py --sweep lanes --condition 2_lanes --pref safety

Options
-------
--sweep       density | lanes                  (default: density)
--condition   empty | normal | crowded         (for density)
              2_lanes | 3_lanes | 4_lanes      (for lanes)
--model       psl | baseline                   (default: psl)
--pref        safety | speed | comfort | uniform | all
--n_eps       number of episodes to run        (default: 3)
--video       save mp4 instead of live window
--fps         video frames per second          (default: 15)
--out_dir     directory for saved videos       (default: checkpoints/exp4/videos)
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────

CKPT_ROOT = "checkpoints/exp4"

SWEEP_CFGS = {
    "density": {
        "empty":   {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 3, "vehicles_count": 15},
        "normal":  {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 3, "vehicles_count": 30},
        "crowded": {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 3, "vehicles_count": 50},
    },
    "lanes": {
        "2_lanes": {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 2, "vehicles_count": 30},
        "3_lanes": {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 3, "vehicles_count": 30},
        "4_lanes": {"duration": 15, "policy_frequency": 2, "simulation_frequency": 15,
                    "lanes_count": 4, "vehicles_count": 30},
    },
}

PSL_EVAL_LAMS = {
    "safety":  np.array([0.90, 0.05, 0.05], dtype=np.float32),
    "speed":   np.array([0.05, 0.90, 0.05], dtype=np.float32),
    "comfort": np.array([0.05, 0.05, 0.90], dtype=np.float32),
    "uniform": np.array([1/3,  1/3,  1/3],  dtype=np.float32),
}

ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

POLICY_CFG = {"obs_dim": 25, "lam_dim": 3, "hidden_dim": 128, "n_actions": 5}


# ── Checkpoint loaders ───────────────────────────────────────────────────────

def load_psl_policy(sweep: str, condition: str) -> torch.nn.Module:
    """Load the latest PSL checkpoint for a given sweep/condition."""
    from models.policy import ConditionedPolicy

    ckpt_dir = os.path.join(CKPT_ROOT, sweep, f"psl_{condition}")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "psl_upd*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No PSL checkpoints found in {ckpt_dir}")

    saved = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    policy = ConditionedPolicy(**POLICY_CFG)

    # PSLTrainer saves: {'step', 'policy_state', 'critic_state', 'policy_opt', 'critic_opt'}
    if "policy_state" in saved:
        policy.load_state_dict(saved["policy_state"])
    elif "policy_state_dict" in saved:
        policy.load_state_dict(saved["policy_state_dict"])
    else:
        policy.load_state_dict(saved)

    policy.eval()
    print(f"Loaded PSL checkpoint: {os.path.basename(ckpts[-1])}")
    return policy


def load_baseline_policy(sweep: str, condition: str, pref: str) -> torch.nn.Module:
    """Load the latest baseline checkpoint for a given sweep/condition/preset."""
    from models.policy import ConditionedPolicy

    ckpt_dir = os.path.join(CKPT_ROOT, sweep, f"baseline_{condition}", pref)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ep*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No baseline checkpoints found in {ckpt_dir}")

    saved = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    policy = ConditionedPolicy(**POLICY_CFG)
    policy.load_state_dict(saved["policy_state_dict"])
    policy.eval()
    print(f"Loaded baseline checkpoint: {os.path.basename(ckpts[-1])}")
    return policy


# ── Single-episode runner ────────────────────────────────────────────────────

def run_episode(env, policy, lam_np, greedy: bool = True):
    """Run one episode with rendering. Returns (mean_costs, actions, total_steps, crashed, truncated)."""
    import envs.objectives as _obj
    _obj.TTC_THRESHOLD = 3.0

    lam_t = torch.tensor(lam_np, dtype=torch.float32)
    obs, _ = env.reset()
    env.render()          # open the window on the first frame
    done = False
    step_costs, actions = [], []
    terminated = False
    truncated = False

    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs.flatten(), dtype=torch.float32)
            logits = policy(obs_t, lam_t)
            if greedy:
                action = int(logits.argmax().item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample())
            obs, cost_vec, terminated, truncated, _ = env.step(action)
            env.render()  # update the window every step
            step_costs.append(np.array(cost_vec, dtype=np.float32))
            actions.append(action)
            done = terminated or truncated

    mean_costs = np.mean(step_costs, axis=0) if step_costs else np.zeros(3)
    crashed = bool(env.unwrapped.vehicle.crashed) if hasattr(env.unwrapped, "vehicle") else False
    return mean_costs, actions, len(actions), crashed, truncated


# ── Video writer ─────────────────────────────────────────────────────────────

def run_episode_video(env, policy, lam_np, fps: int, out_path: str):
    """Run one episode and save frames as mp4."""
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed. Run: pip install opencv-python")
        return

    import envs.objectives as _obj
    _obj.TTC_THRESHOLD = 3.0

    lam_t = torch.tensor(lam_np, dtype=torch.float32)
    obs, _ = env.reset()
    done = False
    frames = []

    with torch.no_grad():
        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            obs_t = torch.tensor(obs.flatten(), dtype=torch.float32)
            logits = policy(obs_t, lam_t)
            action = int(logits.argmax().item())
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    # Final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    if not frames:
        print("  No frames captured.")
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Video saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Simulate trained PSL / baseline policies.")
    p.add_argument("--sweep",     default="density",
                   choices=["density", "lanes"])
    p.add_argument("--condition", default=None,
                   help="Condition key (e.g. 'normal', 'crowded', '3_lanes'). "
                        "Defaults to 'normal' for density, '3_lanes' for lanes.")
    p.add_argument("--model",     default="psl",
                   choices=["psl", "baseline"])
    p.add_argument("--pref",      default="safety",
                   choices=["safety", "speed", "comfort", "uniform", "all"])
    p.add_argument("--n_eps",     type=int, default=0,
                   help="Number of episodes to run per preference. "
                        "0 (default) = loop forever until Ctrl-C.")
    p.add_argument("--duration",  type=int, default=300,
                   help="Max episode length in seconds (default 300). "
                        "Episode ends early if the ego vehicle crashes.")
    p.add_argument("--video",     action="store_true",
                   help="Save mp4 video instead of opening a live window.")
    p.add_argument("--fps",       type=int, default=15)
    p.add_argument("--out_dir",   default=os.path.join(CKPT_ROOT, "videos"))
    return p.parse_args()


def main():
    args = parse_args()

    # Default condition per sweep
    if args.condition is None:
        args.condition = "normal" if args.sweep == "density" else "3_lanes"

    if args.condition not in SWEEP_CFGS[args.sweep]:
        print(f"Unknown condition '{args.condition}' for sweep '{args.sweep}'.")
        print(f"  Available: {list(SWEEP_CFGS[args.sweep].keys())}")
        sys.exit(1)

    env_cfg = dict(SWEEP_CFGS[args.sweep][args.condition])  # copy so we can mutate
    env_cfg["duration"] = args.duration

    # Determine which preferences to run
    prefs = list(PSL_EVAL_LAMS.keys()) if args.pref == "all" else [args.pref]

    for pref in prefs:
        lam_np = PSL_EVAL_LAMS[pref]

        print(f"\n{'='*60}")
        print(f"  Sweep: {args.sweep}   Condition: {args.condition}")
        print(f"  Model: {args.model}   Preference: {pref}   λ = {lam_np}")
        print(f"{'='*60}")

        # Load policy
        try:
            if args.model == "psl":
                policy = load_psl_policy(args.sweep, args.condition)
            else:
                policy = load_baseline_policy(args.sweep, args.condition, pref)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # Build environment
        from envs import MOHighwayEnv

        render_mode = "rgb_array" if args.video else "human"
        env = MOHighwayEnv(config=env_cfg, render_mode=render_mode)

        if args.video:
            os.makedirs(args.out_dir, exist_ok=True)

        n_eps = args.n_eps if (args.n_eps > 0) else (3 if args.video else None)
        ep = 0
        try:
            while n_eps is None or ep < n_eps:
                ep += 1
                ep_label = f"ep{ep:02d}" if n_eps else f"ep{ep}"
                label = f"{args.model}_{args.sweep}_{args.condition}_{pref}_{ep_label}"
                n_str = f"/{n_eps}" if n_eps else " (Ctrl-C to stop)"
                print(f"\n  Episode {ep}{n_str}  ({label})")

                if args.video:
                    out_path = os.path.join(args.out_dir, f"{label}.mp4")
                    run_episode_video(env, policy, lam_np, args.fps, out_path)
                else:
                    mean_costs, actions, steps, crashed, time_limit = run_episode(
                        env, policy, lam_np, greedy=True
                    )
                    survival_s = steps * 0.5  # policy_frequency=2Hz → dt=0.5s
                    end_reason = "CRASHED" if crashed else ("TIME LIMIT" if time_limit else "done")
                    action_counts = {a: actions.count(i) for i, a in enumerate(ACTION_NAMES)}
                    dominant = max(action_counts, key=action_counts.get)
                    print(f"    survival: {survival_s:.1f}s ({steps} steps)  end: {end_reason}")
                    print(f"    f_safety={mean_costs[0]:.3f}  "
                          f"f_speed={mean_costs[1]:.3f}  "
                          f"f_comfort={mean_costs[2]:.3f}")
                    print(f"    dominant action: {dominant}  "
                          f"({action_counts[dominant]/steps*100:.0f}%)" if steps else "")
        except KeyboardInterrupt:
            print(f"\n  Stopped after {ep} episode(s).")

        env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
