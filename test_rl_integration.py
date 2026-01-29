"""
Test RL Integration with Existing Simulation

This script demonstrates how the RL agent integrates with the existing
simulation code with minimal changes.

Usage:
    python test_rl_integration.py
"""

import sys
import simpy
from pathlib import Path

# Add simulation directory to path
sim_dir = Path(__file__).parent
sys.path.insert(0, str(sim_dir))

import config as cfg
import sim_model_RL as model
from reporting import KPI, LOG, update_state
import rl_agent


def test_basic_simulation():
    """Test basic simulation with RL integration"""
    
    print("=" * 70)
    print("Testing RL Integration with Existing Simulation")
    print("=" * 70)
    
    # Test configurations
    test_cases = [
        {
            'name': 'Rule-based (Original)',
            'dispatch_rule': 'RANDOM',
            'use_rl_charging': False
        },
        {
            'name': 'RL Bidding (Dispatch)',
            'dispatch_rule': 'RL_BIDDING',
            'use_rl_charging': False
        },
        {
            'name': 'RL Charging Decision',
            'dispatch_rule': 'RANDOM',
            'use_rl_charging': True
        },
        {
            'name': 'Full RL (Bidding + Charging)',
            'dispatch_rule': 'RL_BIDDING',
            'use_rl_charging': True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'-' * 70}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'-' * 70}")
        
        # Set config
        cfg.DISPATCHING_RULE = test_case['dispatch_rule']
        cfg.USE_RL_CHARGING = test_case['use_rl_charging']
        
        # Reset RL agents
        rl_agent.reset_agents()
        
        # Run simulation
        result = run_single_simulation(test_case['name'])
        results.append(result)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Test Case':<35} {'Flights':<10} {'Delays':<10} {'Energy (kWh)':<15}")
    print("-" * 70)
    
    for res in results:
        print(f"{res['name']:<35} {res['flights']:<10} {res['delays']:<10} {res['energy']:<15.1f}")
    
    print("=" * 70)


def run_single_simulation(test_name: str, num_flights: int = 5):
    """Run a single simulation instance"""
    
    # Create SimPy environment
    env = simpy.Environment()
    kpi = KPI()
    
    # Build infrastructure
    chargers = {}
    for label in model.CHARGER_LABELS:
        chargers[label] = model.ChargerBank(env, label, kpi, capacity=cfg.CHARGER_CAPACITY)
    
    # Create fleets
    fleets = {}
    for kind, size in cfg.FLEET_SIZE.items():
        fleet = model.AMRFleet(env, kind, size, chargers, kpi, rl_mode=False)
        fleets[kind] = fleet
    
    # Create gates
    gates = {}
    for i in range(1, 11):  # 10 gates
        gate_label = f"C{i}"
        gates[gate_label] = simpy.Resource(env, capacity=1)
    
    # Schedule flights
    for i in range(num_flights):
        gate = f"C{(i % 10) + 1}"
        flight_id = f"FLT{i+1:03d}"
        arrival_time = i * 20.0  # Every 20 minutes
        
        env.process(
            model.flight_starter(env, arrival_time, flight_id, gate, gates, fleets, kpi)
        )
    
    # Run simulation
    print(f"Running simulation: {test_name}...")
    print(f"  - Dispatch Rule: {cfg.DISPATCHING_RULE}")
    print(f"  - RL Charging: {cfg.USE_RL_CHARGING}")
    print(f"  - Number of Flights: {num_flights}")
    
    env.run()
    
    # Collect results
    result = {
        'name': test_name,
        'flights': len(kpi.flight_turnaround_times),
        'delays': kpi.flight_delays_count,
        'energy': kpi.total_energy_consumed,
        'avg_turnaround': sum(kpi.flight_turnaround_times) / len(kpi.flight_turnaround_times) if kpi.flight_turnaround_times else 0
    }
    
    print(f"  ✓ Completed!")
    print(f"    - Flights Handled: {result['flights']}")
    print(f"    - Delays: {result['delays']}")
    print(f"    - Total Energy: {result['energy']:.1f} kWh")
    print(f"    - Avg Turnaround: {result['avg_turnaround']:.1f} min")
    
    return result


def test_rl_agent_directly():
    """Test RL agent functions directly"""
    
    print("\n" + "=" * 70)
    print("Testing RL Agent Functions Directly")
    print("=" * 70)
    
    # Test charging agent
    print("\n[Test 1] Charging Agent")
    charging_agent = rl_agent.get_charging_agent(mode='rule')
    
    test_states = [
        {'battery': 0.2, 'ch1_queue': 1, 'ch2_queue': 2, 'desc': 'Low battery, Ch1 shorter'},
        {'battery': 0.6, 'ch1_queue': 0, 'ch2_queue': 0, 'desc': 'High battery, no queue'},
        {'battery': 0.4, 'ch1_queue': 3, 'ch2_queue': 1, 'desc': 'Medium battery, Ch2 shorter'},
    ]
    
    for ts in test_states:
        state = rl_agent.AgentState(
            battery=ts['battery'],
            position=(3, 2),
            total_work_time=100,
            num_tasks=5,
            charger1_queue=ts['ch1_queue'],
            charger2_queue=ts['ch2_queue'],
            time=50.0
        )
        action = charging_agent.select_charging_action(state)
        action_name = ['No Charge', 'Ch1', 'Ch2'][action]
        print(f"  {ts['desc']:<40} → {action_name}")
    
    # Test dispatch agent
    print("\n[Test 2] Dispatch Agent (Bidding)")
    dispatch_agent = rl_agent.get_dispatch_agent(mode='rule')
    
    # Create mock AGVs
    class MockAGV:
        def __init__(self, id, battery, workload):
            self.agv_id = id
            self.soc_percent = battery
            self.position_x = 3
            self.position_y = 2
            self.total_work_time = workload
            self.num_tasks = workload // 20
    
    agvs = [
        MockAGV(1, 0.9, 50),   # High battery, low workload
        MockAGV(2, 0.3, 200),  # Low battery, high workload
        MockAGV(3, 0.7, 100),  # Medium battery, medium workload
    ]
    
    task_info = {
        'gate': 'C3',
        'gate_position': (0, 4),
        'task_type': 'FUEL',
        'required_energy': 10,
        'ch1_queue': 1,
        'ch2_queue': 0,
        'time': 100.0
    }
    
    print(f"\n  Computing bids for task at {task_info['gate']}:")
    for agv in agvs:
        bid = dispatch_agent.compute_bid_score(agv, task_info)
        print(f"    AGV {agv.agv_id} (SOC={agv.soc_percent:.1f}, Work={agv.total_work_time:3d}) → Bid: {bid:.2f}")
    
    selected = dispatch_agent.select_agvs_for_task(agvs, task_info, n=2)
    print(f"\n  Selected AGVs: {[agv.agv_id for agv in selected]}")
    
    print("\n" + "=" * 70)
    print("Direct RL Agent Test Passed!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RL integration')
    parser.add_argument('--test', choices=['agent', 'sim', 'all'], default='all',
                       help='Which test to run')
    parser.add_argument('--flights', type=int, default=3,
                       help='Number of flights for simulation test')
    
    args = parser.parse_args()
    
    if args.test in ['agent', 'all']:
        test_rl_agent_directly()
    
    if args.test in ['sim', 'all']:
        test_basic_simulation()
    
    print("\n✓ All tests completed!")
