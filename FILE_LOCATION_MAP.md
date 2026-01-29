# Airport eGSE Scheduling - RL Simulation

## Overview

This project implements **Reinforcement Learning** for airport ground support equipment (eGSE) scheduling, specifically for **AGV charging decisions**. It includes both baseline DQN and advanced **DR-CVaR-QRDQN** (Domain Randomization + Conditional Value at Risk + Quantile Regression DQN) for robust, risk-averse policies.

---

## Project Structure

```
RL_simulation/
│
├── Core Simulation
│   ├── config.py                 # Configuration (fleet, energy, RL settings)
│   ├── main.py                   # Basic simulation entry point
│   ├── model.py                  # AMR/Fleet/Charger classes (heuristic)
│   ├── sim_model_RL.py           # RL-integrated simulation model
│   ├── routing.py                # Airport map & path calculation
│   └── reporting.py              # State management, KPI, logging
│
├── RL Agents
│   ├── rl_agent.py               # Basic DQN agent (original)
│   └── agents.py                 # DQN + QR-DQN agents (new)
│
├── Training Scripts
│   ├── train_rl.py               # Basic RL training loop
│   ├── train_dqn_wandb.py        # DQN training with W&B logging
│   └── train_dr_cvar_qrdqn_wandb.py  # DR + CVaR + QR-DQN training
│
├── Utilities
│   ├── state_utils.py            # State extraction (raw + normalized)
│   ├── dr_utils.py               # Domain Randomization (BASIC/WEAK/STRONG)
│   ├── wandb_utils.py            # Weights & Biases logging
│   └── benchmark_compare_wandb.py # Heuristic vs RL benchmark
│
├── Tests & Docs
│   ├── test_rl_integration.py    # RL module tests
│   ├── FILE_LOCATION_MAP.md      # This file
│   └── README_PATCH.md           # W&B training guide
│
├── Data/                         # Input data
│   ├── x-SFO-y_gate.csv          # Gate coordinates & flight arrivals
│   ├── flights_sample_3m_SFO.csv # SFO flight data (3 months)
│   └── ...
│
└── Results_TH/                   # Output results
    └── YYYYMMDD_HHMMSS_method/   # Timestamped result folders
        ├── training_curves.png
        ├── training_history.csv
        ├── kpi_amr_utilization.csv
        └── plot_gate_gantt.png
```

---

## RL Framework Comparison

| Feature | Basic DQN | DR-CVaR-QRDQN |
|---------|-----------|---------------|
| **File** | `train_rl.py`, `train_dqn_wandb.py` | `train_dr_cvar_qrdqn_wandb.py` |
| **Agent** | `rl_agent.py` or `agents.py` (DQN) | `agents.py` (QRDQNAgent) |
| **Value Estimation** | Single Q-value | Quantile distribution |
| **Risk Handling** | Risk-neutral | CVaR (risk-averse, lower tail) |
| **Domain Randomization** | None or basic | BASIC/WEAK/STRONG perturbations |
| **W&B Logging** | Optional | Built-in |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install simpy pandas numpy torch matplotlib wandb
```

### 2. Set W&B API Key (optional)

```bash
# Either of these works:
export WANDB_API_KEY="your_key_here"
export wandb_api_key="your_key_here"
```

### 3. Run Basic Simulation (Heuristic)

```bash
python main.py
```

### 4. Run Basic RL Training

```bash
python train_rl.py --episodes 100 --eval_interval 20
```

### 5. Run DQN with W&B

```bash
python train_dqn_wandb.py --episodes 2000 --eval-every 50 --wandb
```

### 6. Run Robust RL (DR + CVaR + QR-DQN)

```bash
python train_dr_cvar_qrdqn_wandb.py \
  --episodes 2000 \
  --train-perturb strong \
  --eval-perturb strong \
  --risk-mode cvar \
  --cvar-alpha 0.2 \
  --wandb
```

### 7. Benchmark Comparison

```bash
python benchmark_compare_wandb.py \
  --dqn-ckpt checkpoints/dqn_charging.pt \
  --qrdqn-ckpt checkpoints/dr_cvar_qrdqn.pt \
  --n-seeds 20 \
  --wandb
```

---

## Key Components

### State Space (7 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `battery_soc` | Battery state of charge (0-100%) |
| 1 | `t_to_charger1` | Travel time to Charger 1 (min) |
| 2 | `t_to_charger2` | Travel time to Charger 2 (min) |
| 3 | `queue_charger1` | Queue length at Charger 1 |
| 4 | `queue_charger2` | Queue length at Charger 2 |
| 5 | `workload` | Cumulative work time (min) |
| 6 | `time_norm` | Normalized simulation time (0-1) |

### Action Space (3 actions)

| Action | Description |
|--------|-------------|
| 0 | No charging (go to depot) |
| 1 | Charge at Charger 1 |
| 2 | Charge at Charger 2 |

### Reward Function

```python
def _calc_charge_reward(state, action, next_state):
    # Penalize low battery without charging
    # Reward appropriate charging decisions
    # Penalize charging when queue is long
    # Consider workload distribution
```

---

## Domain Randomization Levels

| Level | Battery Scale | Charge Power | Travel Power | Service Noise |
|-------|--------------|--------------|--------------|---------------|
| **BASIC** | 1.0 | 1.0 | 1.0 | 0% |
| **WEAK** | 0.85-1.0 | 0.9-1.1 | 0.9-1.1 | 5% |
| **STRONG** | 0.60-1.0 | 0.75-1.25 | 0.75-1.25 | 20% |

---

## Configuration Reference

### config.py Key Settings

```python
# Dispatching Rules
DISPATCHING_RULE = 'FIFO'  # FIFO, RANDOM, LEAST_UTILIZED, BIDDING

# RL Settings
USE_RL_CHARGING = True     # Enable RL-based charging decisions

# Fleet Configuration
FLEET_SIZE = {
    'GPU': 10, 'FUEL': 5, 'WATER': 4,
    'CLEAN': 8, 'CATERING': 8, 'BAGGAGE': 12
}

# Energy Settings
DEFAULT_BATTERY_CAP_KWH = 40.0
CHARGE_TRIGGER_SOC = 0.3   # 30% triggers charging
CHARGE_POWER_KW = 12.2
CHARGER_CAPACITY = 3

# Output
OUTPUT_BASE_DIR = "Results_TH"
```

---

## Results Folder Structure

Each run creates a timestamped folder:

```
Results_TH/
└── 20260129_212050_RL_train_ep100/
    ├── training_curves.png      # Learning curves visualization
    ├── training_history.csv     # Episode-by-episode metrics
    ├── kpi_amr_utilization.csv  # AMR utilization breakdown
    ├── log_amr_events.csv       # Detailed AMR event log
    ├── log_flight_events.csv    # Flight event log
    └── plot_gate_gantt.png      # Gate occupancy Gantt chart
```

---

## File-by-File Reference

| File | Purpose |
|------|---------|
| `config.py` | All configuration variables |
| `routing.py` | Airport map coordinates, path/distance calculation |
| `model.py` | Basic simulation (AMRUnit, ChargerBank, AMRFleet) |
| `sim_model_RL.py` | RL-integrated simulation with decision queue |
| `rl_agent.py` | Original DQN agent for charging decisions |
| `agents.py` | New DQN + QR-DQN implementations |
| `state_utils.py` | State vector extraction (raw + normalized) |
| `dr_utils.py` | Domain randomization (perturbation specs) |
| `wandb_utils.py` | W&B logging utilities |
| `reporting.py` | EventLogger, KPIs, state snapshots |
| `train_rl.py` | Basic training loop |
| `train_dqn_wandb.py` | DQN training with W&B |
| `train_dr_cvar_qrdqn_wandb.py` | Robust RL training |
| `benchmark_compare_wandb.py` | Heuristic vs RL comparison |

---

## Performance Benchmarks

### Basic DQN (100 episodes)

| Metric | Before Training | After Training |
|--------|-----------------|----------------|
| Delay Rate | ~70% | ~10-15% |
| Avg Turnaround | ~300 min | ~100 min |
| Best Episode | - | 3.2% delay rate |

### Expected Improvements with DR-CVaR-QRDQN

- More robust to environment variations
- Risk-averse charging decisions
- Better generalization to unseen scenarios

---

## Data Files

| File | Description |
|------|-------------|
| `x-SFO-y_gate.csv` | Primary flight data with gate assignments |
| `flights_sample_3m_SFO.csv` | 3-month SFO flight data (full) |
| `flights_sample_3m_SFO_DEST.csv` | SFO arrival flights |
| `flights_sample_3m_SFO_ORIGIN.csv` | SFO departure flights |
| `SFO_Gate_and_Stand_Assignment_Information_20251010.csv` | Gate/stand info |
| `time_AMR_manhattan_25kmh_min.csv` | AMR travel time matrix |
