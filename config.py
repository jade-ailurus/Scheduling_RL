# ==============================================================================
# ===== CONFIGURATION - 모든 설정 변수 =====
# ==============================================================================

import random

# ===== 0. LOGGING & DEBUGGING =====
# ENABLE_SNAPSHOT_LOGGING = True   # True로 설정하면 터미널에 상세 스냅샷을 출력합니다.
# ENABLE_SNAPSHOT_PLOTTING = True # True로 설정하면 스냅샷 맵 이미지를 파일로 저장합니다.
ENABLE_SNAPSHOT_LOGGING = False
ENABLE_SNAPSHOT_PLOTTING = False 

# ===== 1. SCENARIO & FILES =====
RND_SEED = 42
random.seed(RND_SEED)
ARRIVAL_CSV = 'Data/x-SFO-y_gate.csv'
OUTPUT_BASE_DIR = "Results_TH"
OUTPUT_DIR = None  # Will be set dynamically by _setup_output_dir()

# ===== 2. SIMULATION TIME & RULES =====
NUM_FLIGHTS = 74
SIM_BUFFER_MIN = 24 * 60
TARGET_TURNAROUND_MINUTES = 60.0

# Dispatching Rule Options:
# - 'FIFO': First In First Out
# - 'RANDOM': Random selection
# - 'LEAST_UTILIZED': Choose least utilized AGV
# - 'BIDDING': Heuristic bidding (SOC + utilization)
#              With USE_RL_CHARGING=True: RL predicts future charging for bidding
# DISPATCHING_RULE = 'BIDDING'  # Use BIDDING with RL charging prediction
DISPATCHING_RULE = 'FIFO'
# DISPATCHING_RULE = 'RANDOM'
# DISPATCHING_RULE = 'LEAST_UTILIZED'

# ===== RL AGENT CONFIGURATION =====
# Enable RL-based charging decision (True) or use rule-based (False)
USE_RL_CHARGING = True  # Set to True to use RL agent for charging decisions

# ===== 3. eGSE FLEET & TASK CONFIG =====
SERVICE_TIMES = {
    'GPU': 0.0,
    'FUEL': 15.0,
    'WATER': 10.0,
    'CLEAN': 25.0,
    'CATERING': 25.0,
    'BAGGAGE_OUT': 20.0,
    'BAGGAGE_IN': 20.0,
}
FLEET_SIZE = {
    'GPU': 10,
    'FUEL': 5,
    'WATER': 4,
    'CLEAN': 8,
    'CATERING': 8,
    'BAGGAGE': 12
}
TOTAL_AMR_FLEET_SIZE = sum(FLEET_SIZE.values())
TASK_TO_FLEET_MAP = {
    'GPU': 'GPU',
    'FUEL': 'FUEL',
    'WATER': 'WATER',
    'CLEAN': 'CLEAN',
    'CATERING': 'CATERING',
    'BAGGAGE_OUT': 'BAGGAGE',
    'BAGGAGE_IN': 'BAGGAGE',
}
REQUIRED_UNITS = {task: 1 for task in SERVICE_TIMES}

# ===== 4. ENERGY & CHARGING =====
DEFAULT_BATTERY_CAP_KWH = 40.0
CHARGE_TRIGGER_SOC = 0.3
CHARGE_POWER_KW = 12.2
CHARGER_CAPACITY = 3
GPU_CONFIG = {
    'BATTERY_CAP_KWH': 150.0,
    'SERVICE_CONSUME_POWER_KW': 30.0
}
DEFAULT_SERVICE_CONSUME_POWER_KW = 10.0
TRAVEL_CONSUME_POWER_KW = 24.4

# ===== 5. MAP & ROUTING =====
AMR_SPEED_KPH = 15.0
MAP_SCALE_FACTOR = 25.0 #(m) for each x and y
AMR_SPEED_DIST_PER_MIN = (AMR_SPEED_KPH * 1000 / 60)
