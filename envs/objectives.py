"""
Objective functions for multi-objective highway navigation.

All functions return a value in [0, 1] where 0 = best, 1 = worst.
LibMOON minimizes objectives, so lower = more desirable.

Three objectives (from the abstract):
  f_safety  : inverse TTC + minimum-distance violation
  f_speed   : deviation from maximum allowable speed
  f_comfort : mean absolute jerk + lateral acceleration
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Highway speed limits (m/s). highway-env default reward_speed_range = [23, 30].
V_MAX = 30.0         # target (maximum allowed) speed
V_MIN = 0.0

# Safety thresholds
TTC_THRESHOLD = 5.0  # seconds — closer than this starts contributing to f_safety
MIN_SAFE_DIST = 10.0 # metres — hard minimum gap (vehicle length ~5m, so 2x)

# Jerk normalisation: comfortable jerk ≈ 0.9 m/s³, harsh ≈ 3–5 m/s³
MAX_JERK = 5.0       # m/s³ — clips the normalised jerk at 1.0

# Lateral acceleration normalisation: comfortable ≈ 0.3g, harsh ≈ 0.8g
MAX_LAT_ACC = 3.0    # m/s²


# --------------------------------------------------------------------------- #
# f_safety
# --------------------------------------------------------------------------- #

def compute_safety(ego, road_vehicles: list, crashed: bool) -> float:
    """
    f_safety ∈ [0, 1].

    Components:
      1. Inverse TTC with the closest vehicle ahead in the same lane.
         iTTC = clamp(TTC_THRESHOLD / TTC, 0, 1)
         When no vehicle is ahead, or the ego is slower, iTTC = 0.
      2. Minimum-distance violation: 1.0 if any vehicle is within MIN_SAFE_DIST,
         else 0.0.
      3. Crash: hard 1.0 if ego.crashed.

    Final value = max(iTTC, dist_violation, crash_flag).
    Using max rather than sum keeps the value in [0, 1] and makes the crash
    penalty dominate.
    """
    if crashed:
        return 1.0

    ego_pos = ego.position[0]   # longitudinal position
    ego_speed = ego.speed
    ego_lane = ego.lane_index   # tuple ('road_id', 'lane_from', 'lane_to')

    # --- iTTC: find the nearest vehicle ahead in the same lane ---
    min_gap = np.inf
    ittc = 0.0

    for v in road_vehicles:
        if v is ego:
            continue
        if v.lane_index != ego_lane:
            continue
        gap = v.position[0] - ego_pos
        if gap <= 0:             # vehicle is behind ego
            continue
        if gap < min_gap:
            min_gap = gap
            rel_speed = ego_speed - v.speed  # positive = closing
            if rel_speed > 0:
                ttc = gap / rel_speed
                ittc = float(np.clip(TTC_THRESHOLD / ttc, 0.0, 1.0))
            else:
                ittc = 0.0      # not closing → no TTC risk

    # --- Minimum-distance violation: any vehicle, any lane ---
    dist_violation = 0.0
    for v in road_vehicles:
        if v is ego:
            continue
        dist = np.linalg.norm(v.position - ego.position)
        if dist < MIN_SAFE_DIST:
            dist_violation = 1.0
            break

    return float(max(ittc, dist_violation))


# --------------------------------------------------------------------------- #
# f_speed
# --------------------------------------------------------------------------- #

def compute_speed(ego_speed: float) -> float:
    """
    f_speed ∈ [0, 1].

    Penalises deviation from V_MAX (the maximum allowable speed).
    A vehicle driving at V_MAX scores 0; a stopped vehicle scores 1.

    We use a one-sided deviation: going faster than V_MAX is not penalised
    here (highway-env physically caps speed, so it rarely happens) but
    going slower is penalised linearly.
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
      1. Longitudinal jerk = |Δa / Δt|, normalised by MAX_JERK.
      2. Lateral acceleration = |steering * speed²|, normalised by MAX_LAT_ACC.
         (centripetal approximation: a_lat ≈ κ * v² where κ ∝ steering)

    Returns the mean of the two normalised components.

    Why dt matters: jerk is a *rate* (m/s³). Without dividing by dt we would
    be computing Δa (m/s²), which is dimensionally wrong and policy-frequency-
    dependent. We pass dt from the wrapper so the value is consistent regardless
    of simulation frequency.
    """
    # Jerk (longitudinal)
    jerk = abs(current_acceleration - prev_acceleration) / max(dt, 1e-6)
    norm_jerk = float(np.clip(jerk / MAX_JERK, 0.0, 1.0))

    # Lateral acceleration (centripetal approximation)
    lat_acc = abs(steering) * (ego_speed ** 2)
    norm_lat_acc = float(np.clip(lat_acc / MAX_LAT_ACC, 0.0, 1.0))

    return float(np.mean([norm_jerk, norm_lat_acc]))
