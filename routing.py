# ==============================================================================
# ===== 지도 좌표와 경로 계산 함수 =====
# ==============================================================================

import numpy as np

from config import MAP_SCALE_FACTOR, AMR_SPEED_DIST_PER_MIN

def _norm_label(s):
    """Normalize label to lowercase string"""
    return str(s).strip().lower()

# ===== 5. MAP & ROUTING (Phase 2) =====

# Node Coordinates
NODE_POS = {
    "C3": (0, 4), "C4": (0, 0),
    "C5": (2, 4), "C6": (2, 0),
    "C7": (4, 4), "C8": (5, 0),
    "C9": (6, 4), "C10": (7, 0),
    "C11": (8, 4),
    "DEPOT": (3, 2),
    "CHARGING_1": (-2, 6),
    "CHARGING_2": (-2, -2),
    # --- Internal routing nodes ---
    "DEPOT_TOP_EXIT": (3, 4),
    "DEPOT_BOTTOM_EXIT": (3, 0),
    "CORRIDOR_BOTTOM": (-1, 0),
    "CORRIDOR_TOP": (-1, 4),
    "PATH_3_5": (3, 5),
    "PATH_3_n1": (3, -1),
    "PATH_n1_6": (-1, 6),
    "PATH_n1_n2": (-1, -2),
}

# Normalize node labels (lowercase)
NODE_POS = {_norm_label(k): v for k, v in NODE_POS.items()}
GATE_LABELS = sorted([k for k in NODE_POS if k.startswith('c')])
CHARGER_LABELS = ['charging_1', 'charging_2']
DEPOT_LABEL = 'depot'

# ==============================================================================
# ===== MAP & ROUTING LOGIC (Phase 2) =====
# ==============================================================================

def _get_path_waypoints(src_label: str, dst_label: str) -> list:
    """
    Main router. Returns a list of (x, y) waypoints from src to dst.
    Uses the user-provided routing logic.
    """
    src = _norm_label(src_label)
    dst = _norm_label(dst_label)
    
    if src == dst:
        return [NODE_POS[src]]

    # --- 1. From DEPOT ---
    if src == DEPOT_LABEL:
        if dst in GATE_LABELS:
            xg, yg = NODE_POS[dst]
            if yg == 4: # Top gates
                return [NODE_POS['depot'], NODE_POS['depot_top_exit'], (xg, 4)]
            else: # Bottom gates
                return [NODE_POS['depot'], NODE_POS['depot_bottom_exit'], (xg, 0)]
        else:
            raise NotImplementedError(f"Path not defined: {src} -> {dst}")

    # --- 2. From GATE ---
    if src in GATE_LABELS:
        xg, yg = NODE_POS[src]
        
        # 2a. Gate to DEPOT
        if dst == DEPOT_LABEL:
            if yg == 4: # Top gates
                return [(xg, 4), NODE_POS['depot_top_exit'], NODE_POS['depot']]
            else: # Bottom gates
                return [(xg, 0), NODE_POS['depot_bottom_exit'], NODE_POS['depot']]
        
        # 2b. Gate to CHARGER
        if dst in CHARGER_LABELS:
            if dst == 'charging_1':
                if yg == 4: # Top gate to Top charger
                    return [(xg, 4), (xg, 6), NODE_POS['charging_1']]
                else: # Bottom gate to Top charger (via corridor)
                    return [(xg, 0), NODE_POS['corridor_bottom'], NODE_POS['corridor_top'], NODE_POS['path_n1_6'], NODE_POS['charging_1']]
            else: # dst == 'charging_2'
                if yg == 0: # Bottom gate to Bottom charger
                    return [(xg, 0), (xg, -2), NODE_POS['charging_2']]
                else: # Top gate to Bottom charger (via corridor)
                    return [(xg, 4), NODE_POS['corridor_top'], NODE_POS['corridor_bottom'], NODE_POS['path_n1_n2'], NODE_POS['charging_2']]

    # --- 3. From CHARGER ---
    if src in CHARGER_LABELS:
        if dst == DEPOT_LABEL:
            if src == 'charging_1':
                return [NODE_POS['charging_1'], (3, 6), (3, 4), NODE_POS['depot']]
            else: # src == 'charging_2'
                return [NODE_POS['charging_2'], (3, -2), (3, 0), NODE_POS['depot']]

    # Fallback / Error
    raise NotImplementedError(f"Path not defined: {src} -> {dst}")

def _calculate_path_distance_and_time(waypoints: list) -> (float, float):
    """Calculates total distance and travel time for a list of waypoints."""
    if not waypoints or len(waypoints) < 2:
        return 0.0, 0.0
    
    path = np.array(waypoints, dtype=float)
    segment_distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_distance = np.sum(segment_distances) * MAP_SCALE_FACTOR 
    
    travel_time_min = total_distance / AMR_SPEED_DIST_PER_MIN
    return total_distance, travel_time_min

def get_position(path: list, t_start: float, t_end: float, t_now: float):
    """
    (Phase 6) Calculates AMR position at t_now along a path.
    User-provided function.
    """
    path = np.array(path, dtype=float)
    if t_now <= t_start:
        return tuple(path[0])
    if t_now >= t_end or (t_end - t_start) == 0:
        return tuple(path[-1])

    segs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumdist = np.insert(np.cumsum(segs), 0, 0)
    total = cumdist[-1]
    
    if total == 0:
        return tuple(path[0])

    frac = (t_now - t_start) / (t_end - t_start)
    target = frac * total
    
    i = np.searchsorted(cumdist, target) - 1
    i = max(0, min(i, len(segs)-1))

    if segs[i] == 0:
         return tuple(path[i])
         
    local_frac = (target - cumdist[i]) / segs[i]
    pos = path[i] + local_frac * (path[i+1] - path[i])
    return tuple(np.round(pos, 2))