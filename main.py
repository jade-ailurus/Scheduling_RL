import simpy
import pandas as pd
from datetime import datetime
import random
import config as cfg
from routing import (NODE_POS, GATE_LABELS, _norm_label, _get_path_waypoints, _calculate_path_distance_and_time)
from model import AMRFleet, ChargerBank, flight_starter
from reporting import KPIs, EventLogger, _setup_output_dir, _plot_gate_gantt, _export_logs, LOG

def parse_arrivals(csv_path: str, num_flights: int) -> (pd.DataFrame, float):
    """
    CSV에서 항공편 도착 데이터를 파싱합니다.
    """
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], errors='coerce')
    df['GATE'] = df['GATE'].astype(str).apply(_norm_label)
    # 맵에 정의된 게이트만 필터링
    valid_gates = set(GATE_LABELS)
    gate_mask = df['GATE'].isin(valid_gates)
    time_mask = df['ARR_TIME'].notna()
    df = df[gate_mask & time_mask].copy()

    if len(df) == 0:
        raise ValueError(f"No valid flights remain after filtering for gates: {valid_gates}")
    
    # 샘플링
    n = min(num_flights, len(df))
    df = df.sample(n=n, random_state=cfg.RND_SEED).sort_values('ARR_TIME').reset_index(drop=True)
    # 시간 정규화 (t=0)
    t0 = df['ARR_TIME'].min()
    df['t_start_min'] = (df['ARR_TIME'] - t0).dt.total_seconds() / 60.0
    t_last_arrival = df['t_start_min'].max()
    
    print(f"[INFO] Loaded {len(df)} flights.")
    print(f"[INFO] Time window (min): 0.0 to {t_last_arrival:.1f}")
    return df, t_last_arrival
    
print(f"\n{'=' * 60}")
print("AIRPORT eGSE SIMULATION")
print(f"Dispatching Rule: {cfg.DISPATCHING_RULE}")
print(f"Fleet Size (Total): {cfg.TOTAL_AMR_FLEET_SIZE}")
print(f"Target Turnaround: {cfg.TARGET_TURNAROUND_MINUTES} min")
print(f"{'=' * 60}\n")

_setup_output_dir()
# 1. 데이터 로드
df_flights, t_last_arrival = parse_arrivals(cfg.ARRIVAL_CSV, cfg.NUM_FLIGHTS)
# 2. 환경 및 KPI 초기화
env = simpy.Environment()
kpi = KPIs(cfg.FLEET_SIZE)
kpi.t_start = env.now
# 3. 리소스 생성
gates = {g: simpy.Resource(env, capacity=1) for g in GATE_LABELS}
chargers = {
    'charging_1': ChargerBank(env, 'charging_1', kpi, cfg.CHARGER_CAPACITY),
    'charging_2': ChargerBank(env, 'charging_2', kpi, cfg.CHARGER_CAPACITY),
}
# 4. Fleet 생성
fleets = {
    kind: AMRFleet(env, kind, size, chargers, kpi)
    for kind, size in cfg.FLEET_SIZE.items()
}
# 5. 항공편 스케줄링
for _, row in df_flights.iterrows():
    name = f"FL{row.name:03d}"
    gate_label = row['GATE']
    start_min = row['t_start_min']
    env.process(flight_starter(env, start_min, name, gate_label, gates, fleets, kpi))
# 6. 시뮬레이션 실행
sim_duration = t_last_arrival + cfg.SIM_BUFFER_MIN
print(f"[INFO] Sim horizon (minutes): {sim_duration:.1f}")
env.run(until=sim_duration)
# 7. 결과 리포트
print(f"[INFO] Simulation complete at t={env.now:.1f}")
kpi.report_summary(fleets, sim_duration)
# 8. 시각화 및 로그 내보내기
_plot_gate_gantt()
_export_logs(kpi, fleets, sim_duration)

print(f"\n{'=' * 60}")
print(f"SIMULATION COMPLETE. Results saved to {cfg.OUTPUT_DIR}/")
print(f"{'=' * 60}")

