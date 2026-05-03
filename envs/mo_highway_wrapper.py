"""
Multi-objective wrapper for highway-env.

Replaces the scalar reward with a 3-vector:
    [f_safety, f_speed, f_comfort]  ∈ [0, 1]³

All three components are to be minimised (lower = better).

Usage:
    env = MOHighwayEnv()
    obs, info = env.reset()
    obs, reward_vec, terminated, truncated, info = env.step(action)
    # reward_vec.shape == (3,)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import highway_env  # noqa: F401 — registers highway-v0 with gymnasium

from envs.objectives import compute_safety, compute_speed, compute_comfort


# Default highway-env config overrides.
# We increase vehicle count and lanes for a richer environment, and turn off
# reward normalisation because we compute our own rewards from scratch.
DEFAULT_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 40,            # episode length in seconds
    "policy_frequency": 2,     # agent acts at 2 Hz → dt = 0.5s
    "simulation_frequency": 15,
    "normalize_reward": False, # we don't use the built-in reward at all
    "reward_speed_range": [23, 30],
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,     # positions relative to ego → stable input range
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
}


class MOHighwayEnv(gym.Wrapper):
    """
    A multi-objective wrapper around highway-v0.

    Parameters
    ----------
    config : dict, optional
        Overrides for DEFAULT_CONFIG. Merged at construction time.
    render_mode : str, optional
        Passed to gym.make. Use "human" for visual, None for headless training.
    """

    # Number of objectives. Stored as a class attribute so models can
    # reference it without instantiating the environment.
    N_OBJECTIVES = 3

    def __init__(self, config: dict | None = None, render_mode=None):
        merged_config = {**DEFAULT_CONFIG, **(config or {})}

        base_env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config=merged_config,
        )
        super().__init__(base_env)

        # dt = 1 / policy_frequency, used for jerk computation
        self._dt = 1.0 / merged_config["policy_frequency"]

        # Acceleration from the previous step, needed to compute jerk.
        # Initialised to 0.0 and reset at the start of each episode.
        self._prev_acceleration: float = 0.0

        # Expose a reward_space so downstream code knows the vector shape.
        # Bounds are [0, 1] for all objectives.
        self.reward_space = spaces.Box(
            low=np.zeros(self.N_OBJECTIVES, dtype=np.float32),
            high=np.ones(self.N_OBJECTIVES, dtype=np.float32),
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------- #
    # Gymnasium API
    # ---------------------------------------------------------------------- #

    def reset(self, **kwargs):
        """Reset the environment and clear accumulated state."""
        self._prev_acceleration = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: int):
        """
        Step the environment and return a vector reward.

        Returns
        -------
        obs : np.ndarray, shape (5, 5)
        reward_vec : np.ndarray, shape (3,)  — [f_safety, f_speed, f_comfort]
        terminated : bool
        truncated : bool
        info : dict  — augmented with 'objectives' key containing a named dict
        """
        obs, _scalar_reward, terminated, truncated, info = self.env.step(action)

        reward_vec, objectives_dict = self._compute_objectives()

        # Augment info with per-objective breakdown for logging/debugging
        info["objectives"] = objectives_dict

        # Update acceleration history for next step's jerk computation
        ego = self.env.unwrapped.vehicle
        self._prev_acceleration = float(ego.action.get("acceleration", 0.0))

        return obs, reward_vec, terminated, truncated, info

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _compute_objectives(self) -> tuple[np.ndarray, dict]:
        """
        Compute the three objective values from the current environment state.

        We reach into the unwrapped environment to access raw vehicle state.
        This is intentional: the normalised observation tensor does not contain
        enough precision to compute accurate TTC or jerk values.

        Returns
        -------
        reward_vec : np.ndarray, shape (3,)
        objectives_dict : dict  — named breakdown for logging
        """
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road_vehicles = unwrapped.road.vehicles

        # --- Safety ---
        f_safety = compute_safety(
            ego=ego,
            road_vehicles=road_vehicles,
            crashed=ego.crashed,
        )

        # --- Speed ---
        f_speed = compute_speed(ego_speed=ego.speed)

        # --- Comfort ---
        current_acc = float(ego.action.get("acceleration", 0.0))
        steering = float(ego.action.get("steering", 0.0))
        f_comfort = compute_comfort(
            current_acceleration=current_acc,
            prev_acceleration=self._prev_acceleration,
            steering=steering,
            ego_speed=ego.speed,
            dt=self._dt,
        )

        reward_vec = np.array([f_safety, f_speed, f_comfort], dtype=np.float32)
        objectives_dict = {
            "f_safety": f_safety,
            "f_speed": f_speed,
            "f_comfort": f_comfort,
        }
        return reward_vec, objectives_dict
