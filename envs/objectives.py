"""
Objective functions for multi-objective highway navigation.

All functions return a value in [0, 1] where 0 = best, 1 = worst.
LibMOON minimizes objectives, so lower = more desirable.

Three objectives:
  f_safety  : inverse TTC + minimum-distance violation
  f_speed   : deviation from maximum allowable speed
  f_comfort : mean absolute jerk + lateral acceleration
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

V_MAX         = 30.0   # m/s — target speed
TTC_THRESHOLD = 5.0    # seconds
MIN_SAFE_DIST = 10.0   # metres
MAX_JERK      = 5.0    # m/s³
MAX_LAT_ACC   = 3.0    # m/s²
WHEELBASE     = 2.5    # metres

W_ITTC = 0.6
W_DIST = 0.4


# --------------------------------------------------------------------------- #
# f_safety
# --------------------------------------------------------------------------- #

def compute_safety(ego, road_vehicles: list, crashed: bool) -> float:
    """
    f_safety ∈ [0, 1].

    Components:
      1. iTTC — worst-case inverse TTC across all front same-lane vehicles.
         Formula: clip((TTC_THRESHOLD - TTC) / TTC_THRESHOLD, 0, 1)
         If not closing (rel_speed <= 0), TTC = inf → iTTC = 0.
         Distance alone handles dangerously small gaps at equal speeds.
      2. dist_violation — continuous penalty for front same-lane vehicles
         within MIN_SAFE_DIST. Grows linearly as gap shrinks.

    Weighted sum gives smooth gradients for learning.
    Crash always returns hard 1.0.
    """
    if crashed:
        return 1.0

    ego_pos  = ego.position[0]
    ego_speed = ego.speed
    ego_lane = ego.lane_index

    ittc = 0.0
    dist_violation = 0.0

    for v in road_vehicles:
        if v is ego:
            continue
        if v.lane_index != ego_lane:
            continue
        gap = v.position[0] - ego_pos
        if gap <= 0:
            continue  # front vehicles only

        # iTTC — worst case across all front vehicles.
        # Smooth: iTTC = clip(TTC_THRESHOLD / TTC, 0, 1).
        # Always positive when closing → continuous gradient at all TTC values
        # rather than zero above the threshold.
        rel_speed = ego_speed - v.speed
        if rel_speed > 0:
            ttc = gap / rel_speed
            candidate_ittc = float(np.clip(TTC_THRESHOLD / ttc, 0.0, 1.0))
        else:
            candidate_ittc = 0.0  # not closing — no TTC risk
        ittc = max(ittc, candidate_ittc)

        # dist_violation — continuous, worst case across front vehicles
        if gap < MIN_SAFE_DIST:
            candidate_dist = float(np.clip(
                (MIN_SAFE_DIST - gap) / MIN_SAFE_DIST, 0.0, 1.0
            ))
            dist_violation = max(dist_violation, candidate_dist)

    return float(np.clip(W_ITTC * ittc + W_DIST * dist_violation, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# f_speed
# --------------------------------------------------------------------------- #

def compute_speed(ego_speed: float) -> float:
    """
    f_speed ∈ [0, 1].
    Linear penalty for driving below V_MAX.
    One-sided — highway-env physically caps speed so overspeed is negligible.
    """
    deviation = max(V_MAX - ego_speed, 0.0)
    return float(np.clip(deviation / V_MAX, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# f_comfort
# --------------------------------------------------------------------------- #

def compute_comfort(
    current_acceleration: float,
    prev_acceleration: float,
    steering: float,
    ego_speed: float,
    dt: float,
) -> float:
    """
    f_comfort ∈ [0, 1].

    Components:
      1. Longitudinal jerk = |Δa / dt|, normalised by MAX_JERK.
      2. Lateral acceleration = (|steering| / WHEELBASE) * v²,
         normalised by MAX_LAT_ACC.
    Mean of both components.
    """
    jerk = abs(current_acceleration - prev_acceleration) / max(dt, 1e-3)
    norm_jerk = float(np.clip(jerk / MAX_JERK, 0.0, 1.0))

    lat_acc = (abs(steering) / WHEELBASE) * (ego_speed ** 2)
    norm_lat_acc = float(np.clip(lat_acc / MAX_LAT_ACC, 0.0, 1.0))

    return float(np.mean([norm_jerk, norm_lat_acc]))