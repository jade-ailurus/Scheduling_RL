# ==============================================================================
# ===== 모든 데이터 수집, 계산, 출력 로직 =====
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import statistics as stats
from collections import defaultdict
import random

import config as cfg
from routing import (
    NODE_POS,
    CHARGER_LABELS,
    _get_path_waypoints,
    _calculate_path_distance_and_time,
)

# ==============================================================================
# ===== GLOBAL LOGGING & STATE =====
# ==============================================================================
class EventLogger:
    """Logs all AMR and Flight events for KPIs and visualization."""
    def __init__(self):
        self.amr_events = []
        self.flight_events = []
        self.state_snapshots = [] # (Phase 6)
    def log_amr(self, t, unit_id, kind, event, **kwargs):
        log = {"time": t, "unit_id": unit_id, "kind": kind, "event": event}
        log.update(kwargs)
        self.amr_events.append(log)
    def log_flight(self, t, flight_id, event, **kwargs):
        log = {"time": t, "flight_id": flight_id, "event": event}
        log.update(kwargs)
        self.flight_events.append(log)
    def log_state_snapshot(self, t, snapshot):
        self.state_snapshots.append({"time": t, "snapshot": snapshot})
# Global instance
LOG = EventLogger()

# ==============================================================================
# ===== KPI TRACKER =====
# ==============================================================================
class KPIs:
    """Tracks and reports all simulation KPIs."""
    def __init__(self, fleet_config):
        self.fleet_config = fleet_config
        self.t_start = 0.0
        self.t_end = 0.0
        # 1. Flight Turnaround
        self.flight_turnaround_times = []
        self.flight_delays_count = 0
        self.flight_delay_minutes = []
        # 2. Gate & Service Waits
        self.gate_wait_times = []
        self.gpu_arrival_wait_times = []
        self.baggage_in_wait_times = []
        # 3. Resource Utilization (Time Tracking)
        self.amr_time_log = {
            (kind, unit_id): {"IDLE": 0.0, "TRAVEL": 0.0, "SERVICE": 0.0, 
                             "CHARGING": 0.0, "Q_CHARGE": 0.0, "Q_TASK": 0.0}
            for kind, size in fleet_config.items()
            for unit_id in range(size)
        }
        self.charger_time_log = {c: {"BUSY": 0.0} for c in CHARGER_LABELS}
        # 4. Travel & Energy
        self.total_travel_distance = 0.0
        self.total_charge_events = 0
        self.total_charge_kwh = 0.0
        self.total_energy_consumed = 0.0
    
    def report_summary(self, fleets: dict, sim_duration_min: float):
        print(f"\n{'-'*25} KPI SUMMARY {'-'*25}")
        self.t_end = sim_duration_min
    
        # 집계: 시뮬레이션 종료 시점의 최종 상태 시간을 추가
        for (kind, unit_id), states in self.amr_time_log.items():
            total_time = sum(states.values())
            time_remaining = max(0, sim_duration_min - total_time)
    
            if time_remaining > 0:
                try:
                    global_id = f"{kind}_{unit_id}"
                    unit = fleets[kind].unit_map[global_id]
                    last_state = unit.time_tracker["state"]
                    states[last_state] += time_remaining
                except (KeyError, AttributeError):
                    states["IDLE"] += time_remaining
                    
        def _stat(data, precision=2):
            if not data:
                return 0, 0.0, 0.0, 0.0
            n = len(data)
            mean = stats.mean(data)
            std = stats.stdev(data) if n > 1 else 0.0
            total = sum(data)
            return (n, round(mean, precision), round(std, precision), round(total, precision))

        # --- 1. Flight & Delay KPIs ---
        print("\n### 1. Flight Turnaround & Delays")
        n, mean, std, _ = _stat(self.flight_turnaround_times)
        print(f"  Flight Turnaround (min):    n={n}, avg={mean}, std={std}")
        n, mean, std, total = _stat(self.flight_delay_minutes)
        print(f"  Turnaround Delays (> {cfg.TARGET_TURNAROUND_MINUTES} min):")
        print(f"    - Total Delayed Flights:  {self.flight_delays_count} (out of {len(self.flight_turnaround_times)})")
        print(f"    - Avg Delay Time (min):   {mean} (for delayed flights)")
        # --- 2. Wait Times (Diagnostics) ---
        print("\n### 2. Service Wait Times (Diagnostics)")
        n, mean, std, total = _stat(self.gate_wait_times)
        print(f"  Avg Gate Wait (min):        {mean} (n={n}, total={total:.1f})")
        n, mean, std, total = _stat(self.gpu_arrival_wait_times)
        print(f"  Avg 'Wait for GPU' (min):   {mean} (n={n}, total={total:.1f})")
        n, mean, std, total = _stat(self.baggage_in_wait_times)
        print(f"  Avg 'Wait for BagOut' (min):{mean} (n={n}, total={total:.1f})")
        # --- 3. Travel & Energy ---
        print("\n### 3. Travel & Energy")
        print(f"  Total Travel Distance:      {self.total_travel_distance:.2f} (meters)")
        print(f"  Total Energy Consumed:      {self.total_energy_consumed:.2f} kWh")
        print(f"  Total Energy Charged:       {self.total_charge_kwh:.2f} kWh")
        print(f"  Total Charge Events:        {self.total_charge_events}")
        # --- 4. Resource Utilization ---
        print("\n### 4. Resource Utilization")
        # 4a. AMR Fleet
        print("  eGSE Fleet Utilization (Avg % of sim time):")
        total_time_by_kind = defaultdict(float)
        state_time_by_kind = defaultdict(lambda: defaultdict(float))

        for (kind, _), log in self.amr_time_log.items():
            for state, time in log.items():
                total_time_by_kind[kind] += time
                state_time_by_kind[kind][state] += time
        
        headers = ["FLEET", "IDLE", "TRAVEL", "SERVICE", "Q_TASK", "CHARGING", "Q_CHARGE"]
        print(f"    {' | '.join(headers)}")
        for kind in self.fleet_config.keys():
            fleet_size = self.fleet_config[kind]
            total_avail_time = fleet_size * sim_duration_min
            if total_avail_time == 0: continue
            
            row = [f"{kind.upper():<7}"]
            for state in headers[1:]:
                pct = (state_time_by_kind[kind][state] / total_avail_time) * 100
                row.append(f"{pct: >6.1f}%")
            print(" | ".join(row))

        # 4b. Chargers
        print("\n  Charger Utilization:")
        for charger in CHARGER_LABELS:
            busy_time = self.charger_time_log[charger]["BUSY"]
            util_pct = (busy_time / sim_duration_min) * 100
            print(f"    - {charger}: {util_pct:.2f}% busy")
            
        print(f"{'-'*60}\n")

def plot_simulation_snapshot(t_now, amr_states, charger_queues):
    fig, ax = plt.subplots(figsize=(14, 10)) 
    # Background Areas (gray zones)
    ax.fill_between([-4, -2], -3, 7, color='0.9', alpha=0.6, label='Ramp Area')
    ax.fill_between([-2, 8], 0, 4, color='0.9', alpha=0.6)
    # Depot Line (x=3)
    ax.vlines(x=3, ymin=0, ymax=4, color='blue', linestyle='--', linewidth=2, label='Depot Line')
    # Corridor Line (x=-1)
    ax.vlines(x=-1,  ymin=0, ymax=4, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Corridor')
    # Possible Route Line 
    ax.vlines(x=-1,  ymin=4, ymax=6, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=-1,  ymin=-2, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=0,  ymin=-1, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=0,  ymin=4, ymax=5, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=2,  ymin=-1, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=2,  ymin=4, ymax=5, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=5,  ymin=-1, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=4,  ymin=4, ymax=5, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=7,  ymin=-1, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=6,  ymin=4, ymax=5, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=3,  ymin=4, ymax=6, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=3,  ymin=-2, ymax=0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.vlines(x=8,  ymin=4, ymax=5, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=-2,  xmin=-2, xmax=-1, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=6,   xmin=-2, xmax=-1, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=-1,   xmin=-1, xmax=7, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=5,   xmin=-1, xmax=8, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=6,   xmin=-2, xmax=3, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')
    ax.hlines(y=-2,   xmin=-2, xmax=3, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Possible Route')

    for name, (x, y) in NODE_POS.items():
        color = 'gray'
        s = 10 
        if name.startswith('cs'):
            color = 'yellow'
            s = 12
        elif name.startswith('corridor'):
            color = 'green'
            s = 12
        elif name.startswith('path'):
            color = 'orange'
            s = 12
        elif name.startswith('depot'):
            color = 'blue'
            s = 12
        else:
            color = 'red'
            s = 12
            
        ax.plot(x, y, 'o', markersize=s, color=color, alpha=0.6)
        ax.text(x, y + 0.2, name, ha='center', fontsize=11, fontweight='bold')

    # 충전소 큐 상태 표시
    for name, (x, y) in NODE_POS.items():
        if name in charger_queues:
            q_len = charger_queues[name]
            if q_len > 0:
                ax.text(x, y - 0.4, f"QUEUE: {q_len}", ha='center', fontsize=12, 
                        color='white', backgroundcolor='red', fontweight='bold')

    # AMR 유닛
    kind_colors = {'GPU': 'orange', 'FUEL': 'black', 'WATER': 'cyan', 'CLEAN': 'purple', 'CATERING': 'brown', 'BAGGAGE': 'magenta'}
    
    # Step 3a: AMR을 위치별로 그룹화합니다.
    location_groups = defaultdict(list)
    for amr in amr_states:
        # (x, y) 좌표가 (0,0) (초기화 오류)이 아닌 유효한 위치의 AMR만 그룹화
        if amr['location_label'] in NODE_POS:
            location_groups[amr['location_label']].append(amr)
    
    # Step 3b: 위치 그룹별로 루프를 돌며 수직으로 쌓습니다.
    y_spacing = 0.4  # AMR 아이콘 간의 수직 간격
    for location_label, amr_list in location_groups.items():
        (x_base, y_base) = NODE_POS[location_label]
        n = len(amr_list) # 이 위치에 있는 AMR의 총 개수
        # 스택이 y_base를 중심으로 위아래로 쌓이도록 시작점 계산
        y_start = y_base - ((n - 1) / 2.0) * y_spacing

        for i, amr in enumerate(amr_list):
            # X좌표는 고정, Y좌표만 스택으로 쌓기
            x_plot = x_base
            y_plot = y_start + (i * y_spacing)
            color = kind_colors.get(amr['kind'], 'gray')
    
            # AMR marker/ text
            ax.plot(x_plot, y_plot, 's', markersize=10, color=color, markeredgecolor='black', label=amr['kind'])
            ax.text(x_plot, y_plot - 0.2, f"{amr['global_id']} ({amr['soc_percent']:.0%})", 
                    ha='center', fontsize=9, color=color, fontweight='medium')
    
    ax.set_title(f"t = {t_now:.2f} min", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-4, 9)
    ax.set_ylim(-3, 7)
    ax.set_xticks(np.arange(-4, 10, 1)) 
    ax.set_yticks(np.arange(-3, 8, 1))  
    ax.grid(True, linestyle=':', alpha=0.9, which='both') 
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11, ncol = 3)
    t_int = int(t_now * 100)
    filename = os.path.join(cfg.OUTPUT_DIR, f"snapshot_t_{t_int:08d}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def log_snapshot_details(t_now, amr_states, charger_queues, fleets):
    """
    현재 시뮬레이션 상태의 상세 텍스트 로그를 생성합니다.
    """
    log_lines = []
    log_lines.append(f"\n{'='*60}")
    log_lines.append(f"===== SIMULATION SNAPSHOT (Time: {t_now:.2f} min) =====")
    log_lines.append(f"{'='*60}")

    # 1. 충전소 상태 (요청하신 'queue in each charging station')
    log_lines.append("### Charger Status ###")
    for name, q_len in charger_queues.items():
        log_lines.append(f"  - {name}: {q_len} unit(s) in queue")
    log_lines.append("")

    # 2. AMR 상태 (SOC, Location, Distance to Charger)
    log_lines.append("### AMR Fleet Status ###")
    
    # 'fleets'에서 Charger 객체에 접근 (약간의 트릭)
    chargers = list(fleets.values())[0].chargers if fleets else {}

    for amr in amr_states:
        # 충전소까지의 거리 계산
        dist_to_charge = 9999.0
        best_charger = "N/A"
        
        if chargers:
            # _find_shortestQ_charger 로직을 여기서 간단히 흉내 냅니다.
            options = []
            for name, bank in chargers.items():
                try:
                    q = len(bank.res.queue)
                    waypoints = _get_path_waypoints(amr['location_label'], name)
                    dist, time = _calculate_path_distance_and_time(waypoints)
                    options.append((q, time, name, dist))
                except NotImplementedError:
                    continue
            
            if options:
                options.sort(key=lambda x: (x[0], x[1])) # 큐, 시간 순 정렬
                best_charger = options[0][2]
                dist_to_charge = options[0][3]

        log_lines.append(
            f"  - {amr['global_id']:<12} | "
            f"State: {amr['state']:<8} | "
            f"SOC: {amr['soc_percent']:>6.1%} | "
            f"Loc: {amr['location_label']:<10} | "
            f"Dist to Charge ({best_charger}): {dist_to_charge:.1f} m"
        )
    
    log_lines.append(f"{'='*60}\n")
    
    print("\n".join(log_lines))
    
# ==============================================================================
# ===== STATE & REPORTING  =====
# ==============================================================================

def update_state(env, trigger_event: str, kpi, fleets: dict):
    """ Captures a global snapshot, logs details, and plots map."""

    if not cfg.ENABLE_SNAPSHOT_LOGGING and not cfg.ENABLE_SNAPSHOT_PLOTTING:
        return
    
    t_now = env.now

    amr_states = []
    for kind, fleet in fleets.items():
        for unit in fleet.units:
            location_xy = NODE_POS.get(unit.location, (0,0)) # (0,0)을 기본값으로

            amr_states.append({
                "global_id": unit.global_id,
                "kind": unit.kind,
                "soc_percent": unit.soc_percent,
                "location_label": unit.location,
                "location_xy": location_xy,
                "state": unit.time_tracker["state"],
                "total_work_time": unit.total_work_time,
            })

    # charger_queues 정보 수집
    charger_queues = {}
    if fleets:
        # fleets 딕셔너리에서 Charger 객체를 가져옴
        chargers = list(fleets.values())[0].chargers
        for name, bank in chargers.items():
            charger_queues[name] = len(bank.res.queue)

    try:
        if cfg.ENABLE_SNAPSHOT_LOGGING:
            log_snapshot_details(t_now, amr_states, charger_queues, fleets)

        if cfg.ENABLE_SNAPSHOT_PLOTTING:
            plot_simulation_snapshot(t_now, amr_states, charger_queues)

    except Exception as e:
        print(f"[ERROR] Failed to generate snapshot at t={t_now}: {e}")

    snapshot = {
        "event": trigger_event,
        "amr_states": amr_states,
        "charger_states": charger_queues 
    }
    LOG.log_state_snapshot(t_now, snapshot)
    
    # Return snapshot for RL usage
    return snapshot
    
def _setup_output_dir():
    """Creates the output directory."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

def _export_logs(kpi, fleets: dict, sim_duration: float):
    """Exports raw event logs and KPI summaries to CSV."""
    if LOG.amr_events:
        pd.DataFrame(LOG.amr_events).to_csv(
            f"{cfg.OUTPUT_DIR}/log_amr_events.csv", index=False
        )
    if LOG.flight_events:
        pd.DataFrame(LOG.flight_events).to_csv(
            f"{cfg.OUTPUT_DIR}/log_flight_events.csv", index=False
        )
        
    kpi_rows = []
    for (kind, unit_id), states in kpi.amr_time_log.items():
        row = {"kind": kind, "unit_id": unit_id}
        row.update(states)
        kpi_rows.append(row)
    pd.DataFrame(kpi_rows).to_csv(
        f"{cfg.OUTPUT_DIR}/kpi_amr_utilization.csv", index=False
    )
    
def _plot_gate_gantt():
    """Plots gate occupancy Gantt chart."""
    df = pd.DataFrame(LOG.flight_events)
    
    if df.empty:
        print("[VIZ] No gate logs to plot. (DataFrame is empty)")
        return
    
    df_gate = df[df['event'].isin(['gate_start', 'gate_end'])].pivot(
        index='flight_id', columns='event', values='time'
    ).reset_index()
    df_gate = df_gate.merge(
        df[df['event'] == 'gate_start'][['flight_id', 'gate']],
        on='flight_id'
    )
    
    if df_gate.empty:
        print("[VIZ] No gate logs to plot.")
        return

    gates_used = sorted(df_gate['gate'].unique())
    color_map = {g: plt.cm.tab20(i % 20) for i, g in enumerate(gates_used)}
    
    fig, ax = plt.subplots(figsize=(15, max(4, 0.4 * len(gates_used))))
    
    yticks, ylabels = [], []
    y = 0
    for gate in gates_used:
        sub = df_gate[df_gate['gate'] == gate].sort_values('gate_start')
        for _, r in sub.iterrows():
            if pd.isna(r['gate_start']) or pd.isna(r['gate_end']):
                continue
            duration = r['gate_end'] - r['gate_start']
            ax.barh(y, duration, left=r['gate_start'], height=0.5,
                    color=color_map[gate], edgecolor='black', alpha=0.7)
            ax.text(r['gate_start'] + duration / 2, y, r['flight_id'],
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
            
        yticks.append(y)
        ylabels.append(gate)
        y += 1

    ax.set_xlabel("Time (minutes)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title(f"Gate Occupancy (Gantt) - Rule: {cfg.DISPATCHING_RULE}")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/plot_gate_gantt.png", dpi=200)
    plt.close(fig)
    print(f"[VIZ] Saved gate Gantt → {cfg.OUTPUT_DIR}/plot_gate_gantt.png")