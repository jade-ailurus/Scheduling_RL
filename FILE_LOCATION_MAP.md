# 핵심 요약 내용의 파일 위치 맵

## 📁 프로젝트 구조

```
RL_simulation/
├── FILE_LOCATION_MAP.md          # 이 파일
└── RL_simulation/
    ├── config.py                 # 설정 변수 (fleet, energy, RL 설정)
    ├── main.py                   # 시뮬레이션 메인 루틴
    ├── model.py                  # 기본 AMR/Fleet/Charger 클래스
    ├── reporting.py              # 상태 관리 & KPI & 로깅
    ├── routing.py                # 공항 맵 & 경로 계산
    ├── rl_agent.py               # DQN 기반 RL 에이전트
    ├── sim_model_RL.py           # RL 통합 시뮬레이션 모델
    ├── test_rl_integration.py    # RL 모듈 테스트
    ├── Data/                     # 입력 데이터
    │   ├── flights_sample_3m_SFO.csv
    │   ├── flights_sample_3m_SFO_DEST.csv
    │   ├── flights_sample_3m_SFO_ORIGIN.csv
    │   ├── SFO_Gate_and_Stand_Assignment_Information_20251010.csv
    │   ├── time_AMR_manhattan_25kmh_min.csv
    │   └── x-SFO-y_gate.csv
    └── Results_TH/               # 시뮬레이션 결과
        ├── kpi_amr_utilization.csv
        ├── log_amr_events.csv
        ├── log_flight_events.csv
        └── plot_gate_gantt.png
```

---

## 📍 주요 내용별 파일 위치

### 1️⃣ STATE 정의 (상태 정보 구조)
**파일**: `reporting.py`
- **함수**: `update_state()` (line 315)
- **내용**:
  - AMR 상태 수집: `global_id`, `kind`, `soc_percent`, `location_label`, `state`, `total_work_time`
  - Charger 상태 수집: 큐 길이
  - snapshot 생성 및 LOG에 저장

---

### 2️⃣ ACTION 정의 (Dispatch Rule)
**파일**: `model.py`
- **함수**: `_select_units_by_rule()` (line 214)
- **내용**:
  - FIFO 규칙
  - RANDOM 규칙
  - LEAST_UTILIZED 규칙
  - BIDDING 규칙 (battery SOC + utilization 기반)

**설정**: `config.py` (line 23~)
```python
DISPATCHING_RULE = 'FIFO'
# DISPATCHING_RULE = 'RANDOM'
# DISPATCHING_RULE = 'BIDDING'
```

---

### 3️⃣ RL 에이전트 (충전 결정)
**파일**: `rl_agent.py`
- **클래스**: `DQN` - 신경망 (128→128 dense layers)
- **클래스**: `ChargingAgent` - RL 에이전트
  - **Action space**: 0 (충전 안 함), 1 (Charger 1), 2 (Charger 2)
  - **State vector**: [SOC %, 충전소1 거리, 충전소2 거리, 대기열1, 대기열2, 작업량, 시간]
  - **Experience replay**: 10,000 capacity
  - **Epsilon decay**: 1.0 → 0.01

**설정**: `config.py`
```python
USE_RL_CHARGING = True  # RL 기반 충전 결정 사용
```

---

### 4️⃣ REQUEST-RELEASE 사이클
**파일**: `model.py`

#### REQUEST (유닛 할당)
- **함수**: `request_units()` (line 246)
- **내용**:
  - 사용 가능한 유닛 찾기 (`_get_eligible_units()`)
  - dispatch rule 적용 (`_select_units_by_rule()`)
  - 유닛을 task에 할당

#### RELEASE (유닛 반환)
- **함수**: `release_units()` (line 273)
- **내용**:
  - Task 완료 후 state update ("amr_task_end" trigger)
  - 필요시 charging
  - Depot으로 복귀
  - Available 상태로 변경

---

### 5️⃣ 시간 전파 (Time Propagation)
**파일**: `model.py`

#### Travel (이동)
- **함수**: `_travel()` (line 102)
- **코드**:
  ```python
  unit.consume_energy(travel_time, cfg.TRAVEL_CONSUME_POWER_KW, self.kpi)
  yield self.env.timeout(travel_time)  # ← 시간 진행
  ```

#### Service (서비스)
- **함수**: `_service()` (line 130)
- **코드**:
  ```python
  unit.consume_energy(duration_min, cfg.DEFAULT_SERVICE_CONSUME_POWER_KW, self.kpi)
  yield self.env.timeout(duration_min)  # ← 시간 진행
  ```

#### Charging (충전)
- **함수**: `_charge()` (line 145)
- **코드**:
  ```python
  hours_to_charge = need_kwh / CHARGE_POWER_KW
  duration_min = hours_to_charge * 60.0
  yield self.env.timeout(duration_min)  # ← 시간 진행
  unit.soc_kwh = unit.capacity_kwh      # ← SoC 회복
  ```

---

### 6️⃣ 배터리 SoC 전파 (Battery Propagation)
**파일**: `model.py`

#### Energy Consumption
- **함수**: `consume_energy()` (line 50, AMRUnit 클래스)
- **내용**:
  ```python
  def consume_energy(self, duration_min: float, power_kw: float, kpi):
      used_kwh = power_kw * (duration_min / 60.0)
      self.soc_kwh = max(0.0, self.soc_kwh - used_kwh)  # ← 즉시 감소
      kpi.total_energy_consumed += used_kwh
  ```

#### Energy Charging
- **파일**: `model.py`, `_charge()` 함수
- **코드**:
  ```python
  unit.soc_kwh = unit.capacity_kwh  # ← SOC 100%로 회복
  kpi.total_charge_kwh += need_kwh
  ```

---

### 7️⃣ STATE UPDATE 트리거 포인트
**파일**: `model.py`

#### Flight Arrival 시점
- **함수**: `flight_process()` (line 388)
- **코드**: `update_state(env, "flight_arrival", kpi, fleets)` (line 404)
- **역할**: 비행기가 gate에 도착했을 때 state snapshot 생성

#### Task Completion 시점
- **함수**: `_unit_return_logic()` 내부 (line 280)
- **코드**: `update_state(self.env, "amr_task_end", self.kpi, all_fleets)`
- **역할**: AMR이 task를 완료했을 때 state snapshot 생성

---

### 8️⃣ Energy & Charging 관련 설정
**파일**: `config.py`

```python
# Battery Capacity
DEFAULT_BATTERY_CAP_KWH = 40.0  # 일반 AMR
GPU_CONFIG = {'BATTERY_CAP_KWH': 150.0, ...}  # GPU AMR

# Energy Consumption
TRAVEL_CONSUME_POWER_KW = 24.4  # 이동 중 소비
DEFAULT_SERVICE_CONSUME_POWER_KW = 10.0  # 서비스 중 소비
GPU_CONFIG['SERVICE_CONSUME_POWER_KW'] = 30.0  # GPU 서비스 중 소비

# Charging
CHARGE_TRIGGER_SOC = 0.3  # 30% 이하면 충전 시작
CHARGE_POWER_KW = 12.2  # 충전 속도
CHARGER_CAPACITY = 3  # 동시 충전 가능 수
```

---

### 9️⃣ 비행기 프로세스
**파일**: `model.py`

#### Main Flight Process
- **함수**: `flight_process()` (line 388)
- **단계**:
  1. Gate 할당 대기
  2. Flight arrival state update
  3. GPU & OTHER tasks 병렬 시작
  4. GPU process 실행
  5. OTHER tasks 완료 대기
  6. GPU unit 반환
  7. Gate 해제

#### Flight Starter
- **함수**: `flight_starter()` (line 485)
- **역할**: 특정 시간에 flight_process 스케줄링

---

### 🔟 KPI & Reporting
**파일**: `reporting.py`

- **클래스**: `KPIs` (line 38)
- **추적 항목**:
  - Flight turnaround time
  - Flight delays
  - Gate wait times
  - GPU arrival wait times
  - Total travel distance
  - Total energy consumed
  - AMR utilization (시간 기반)
  - Charger utilization

---

## 📊 파일별 역할 요약

| 파일 | 역할 |
|------|------|
| `config.py` | 모든 설정 변수 (fleet 구성, 에너지, 충전, RL 설정) |
| `routing.py` | 공항 맵 좌표, 경로/거리/시간 계산 |
| `model.py` | 기본 시뮬레이션 (AMRUnit, ChargerBank, AMRFleet, flight_process) |
| `sim_model_RL.py` | RL 통합 시뮬레이션 (model.py 확장, RL 에이전트 연동) |
| `rl_agent.py` | DQN 에이전트 (충전 결정 학습) |
| `reporting.py` | EventLogger, KPIs, update_state(), 결과 출력 |
| `main.py` | 시뮬레이션 실행 진입점 |
| `test_rl_integration.py` | RL 모듈 단위 테스트 |

---

## 🎯 RL 통합 포인트

### 1. State 받기
- **파일**: `reporting.py`의 `update_state()` 함수
- **위치**: snapshot 생성 후 return

### 2. Action 주기 (충전 결정)
- **파일**: `sim_model_RL.py`
- **내용**: `rl_agent.select_action(state)`으로 충전 여부/위치 결정

### 3. Reward 계산
- **파일**: `sim_model_RL.py`
- **위치**: task completion 후 state 변화 기반 implicit learning

---

## 📂 Data 폴더 설명

| 파일 | 내용 |
|------|------|
| `flights_sample_3m_SFO.csv` | SFO 3개월 항공편 데이터 (전체) |
| `flights_sample_3m_SFO_DEST.csv` | SFO 도착 항공편 |
| `flights_sample_3m_SFO_ORIGIN.csv` | SFO 출발 항공편 |
| `SFO_Gate_and_Stand_Assignment_Information_20251010.csv` | 게이트/스탠드 할당 정보 |
| `time_AMR_manhattan_25kmh_min.csv` | AMR 이동 시간 매트릭스 |
| `x-SFO-y_gate.csv` | 게이트 좌표 정보 |
