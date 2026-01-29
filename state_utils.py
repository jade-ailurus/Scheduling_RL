"""
state_utils.py

State extraction for charging decisions.

The simulator snapshots provide:
- per-AMR SoC and location
- per-charger queue lengths
Training scripts can transform these into fixed-length vectors.

We export both:
- raw state (matches the repo's reward helper assumptions)
- normalized state (better for neural networks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class StateVectors:
    raw: np.ndarray         # shape [7]
    normalized: np.ndarray  # shape [7]


def extract_state(
    snapshot: Dict[str, Any],
    unit_id: int,
    env_now: float,
    sim_duration_min: float,
    cfg_module: Any,
    routing_module: Any,
) -> StateVectors:
    """
    Returns raw + normalized state vectors.

    Raw format (7 dims):
        [0] battery_soc_percent (0..100)
        [1] t_to_charger1_min
        [2] t_to_charger2_min
        [3] q_len_charger1
        [4] q_len_charger2
        [5] unit_workload_min (cumulative, proxy)
        [6] time_norm (0..1)

    Normalization:
        soc: /100
        travel times: /max_travel_min (heuristic constant)
        queues: /charger_capacity
        workload: /workload_scale_min (heuristic constant)
        time_norm: unchanged
    """
    amr = snapshot["amr_states"][unit_id]
    soc = float(amr.get("soc_percent", 0.0))
    loc = amr.get("location_label", "DEPOT")

    # Charger meta
    q1 = float(snapshot["charger_states"][1].get("queue_length", 0))
    q2 = float(snapshot["charger_states"][2].get("queue_length", 0))

    # Route-based travel time to chargers (minutes)
    ch1_loc = getattr(cfg_module, "CHARGER_1_LOCATION", "C1")
    ch2_loc = getattr(cfg_module, "CHARGER_2_LOCATION", "C10")

    # routing_module expected to expose _get_path_waypoints and _calculate_path_distance_and_time
    wps1 = routing_module._get_path_waypoints(loc, ch1_loc)
    dist1, t1 = routing_module._calculate_path_distance_and_time(wps1)
    wps2 = routing_module._get_path_waypoints(loc, ch2_loc)
    dist2, t2 = routing_module._calculate_path_distance_and_time(wps2)

    # Workload proxy: we try to read from snapshot (if present), else 0.
    # If you want a better proxy, attach per-unit workload into the snapshot in reporting.py.
    workload = float(amr.get("total_work_time", 0.0))

    time_norm = float(env_now) / float(sim_duration_min) if sim_duration_min > 0 else 0.0

    raw = np.asarray([soc, float(t1), float(t2), q1, q2, workload, time_norm], dtype=np.float32)

    # --- normalize ---
    charger_cap = float(getattr(cfg_module, "CHARGER_CAPACITY", 1))
    max_travel_min = 60.0  # heuristic scale (1h); adjust if your layout is bigger
    workload_scale_min = 8.0 * 60.0  # one shift (8h)

    norm = np.asarray(
        [
            soc / 100.0,
            float(t1) / max_travel_min,
            float(t2) / max_travel_min,
            q1 / charger_cap,
            q2 / charger_cap,
            workload / workload_scale_min,
            time_norm,
        ],
        dtype=np.float32,
    )

    # Clip to a reasonable range (prevents blow-ups if mis-scaled)
    norm = np.clip(norm, -5.0, 5.0)

    return StateVectors(raw=raw, normalized=norm)
