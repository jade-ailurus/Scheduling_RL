"""
RL Training Script for Airport AGV Charging Decisions

Usage:
    python train_rl.py --episodes 100 --eval_interval 10
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict with PyTorch

import argparse
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

import config as cfg
from routing import NODE_POS, GATE_LABELS, _norm_label, _get_path_waypoints, _calculate_path_distance_and_time
from sim_model_RL import AMRFleet, ChargerBank, flight_starter
from reporting import KPIs, EventLogger, _setup_output_dir, _plot_gate_gantt, _export_logs, LOG
from rl_agent import get_charging_agent, reset_agent


def parse_arrivals(csv_path: str, num_flights: int):
    """Parse flight arrival data from CSV."""
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], errors='coerce')
    df['GATE'] = df['GATE'].astype(str).apply(_norm_label)

    valid_gates = set(GATE_LABELS)
    gate_mask = df['GATE'].isin(valid_gates)
    time_mask = df['ARR_TIME'].notna()
    df = df[gate_mask & time_mask].copy()

    if len(df) == 0:
        raise ValueError(f"No valid flights remain after filtering for gates: {valid_gates}")

    n = min(num_flights, len(df))
    df = df.sample(n=n, random_state=cfg.RND_SEED).sort_values('ARR_TIME').reset_index(drop=True)

    t0 = df['ARR_TIME'].min()
    df['t_start_min'] = (df['ARR_TIME'] - t0).dt.total_seconds() / 60.0
    t_last_arrival = df['t_start_min'].max()

    return df, t_last_arrival


def run_episode(episode_num: int, df_flights: pd.DataFrame, t_last_arrival: float,
                train_mode: bool = True) -> dict:
    """
    Run a single simulation episode.

    Args:
        episode_num: Episode number
        df_flights: Flight schedule DataFrame
        t_last_arrival: Last arrival time
        train_mode: If True, train the RL agent

    Returns:
        dict: Episode metrics
    """
    # Reset logging
    LOG.amr_events = []
    LOG.flight_events = []
    LOG.state_snapshots = []

    # Initialize environment
    env = simpy.Environment()
    kpi = KPIs(cfg.FLEET_SIZE)
    kpi.t_start = env.now

    # Create resources
    gates = {g: simpy.Resource(env, capacity=1) for g in GATE_LABELS}
    chargers = {
        'charging_1': ChargerBank(env, 'charging_1', kpi, cfg.CHARGER_CAPACITY),
        'charging_2': ChargerBank(env, 'charging_2', kpi, cfg.CHARGER_CAPACITY),
    }

    # Create fleets
    fleets = {
        kind: AMRFleet(env, kind, size, chargers, kpi)
        for kind, size in cfg.FLEET_SIZE.items()
    }

    # Get RL agent
    agent = get_charging_agent()

    # Track previous state/action for each unit (for experience replay)
    unit_history = {}  # {unit_id: {'state': ..., 'action': ...}}
    collected_transitions = []  # Store all transitions for batch training

    # Schedule flights
    for _, row in df_flights.iterrows():
        name = f"FL{row.name:03d}"
        gate_label = row['GATE']
        start_min = row['t_start_min']
        env.process(flight_starter(env, start_min, name, gate_label, gates, fleets, kpi))

    # Run simulation with RL decision handling
    sim_duration = t_last_arrival + cfg.SIM_BUFFER_MIN

    # Process decision queue during simulation
    def process_decisions():
        while True:
            yield env.timeout(0.1)

            for fleet in fleets.values():
                while fleet.decision_queue:
                    decision = fleet.decision_queue.pop(0)

                    if decision['type'] == 'charging':
                        snapshot = decision['snapshot']
                        state_vec = decision['state_vector']
                        unit_id = decision['unit_id']
                        unit = decision['unit']
                        action_event = decision['action_event']

                        # Get action from RL agent
                        action = agent.select_action(snapshot, unit_id)

                        # Store experience for training
                        if train_mode:
                            # If we have previous state for this unit, create transition
                            if unit_id in unit_history:
                                prev = unit_history[unit_id]
                                prev_state = prev['state']
                                prev_action = prev['action']

                                # Calculate reward based on what happened
                                reward = fleet._calc_charge_reward(prev_state, prev_action, state_vec)

                                # Store transition
                                collected_transitions.append((
                                    prev_state,
                                    prev_action,
                                    reward,
                                    state_vec,
                                    False  # not done
                                ))

                            # Save current state/action for next transition
                            unit_history[unit_id] = {
                                'state': state_vec.copy(),
                                'action': action
                            }

                        # Trigger the action
                        action_event.succeed(action)

    env.process(process_decisions())
    env.run(until=sim_duration)

    # Train agent after episode
    if train_mode and collected_transitions:
        # Store all collected transitions
        for transition in collected_transitions:
            agent.store_transition(*transition)

        # Train multiple times
        num_trains = max(10, len(collected_transitions) // 4)
        for _ in range(num_trains):
            agent.train()

        # Update target network periodically
        if episode_num % 5 == 0:
            agent.update_target_network()

        print(f"  [Train] {len(collected_transitions)} transitions, buffer={len(agent.replay_buffer)}", end="")

    # Collect metrics
    metrics = {
        'episode': episode_num,
        'avg_turnaround': np.mean(kpi.flight_turnaround_times) if kpi.flight_turnaround_times else 0,
        'delayed_flights': kpi.flight_delays_count,
        'total_delays': len(kpi.flight_turnaround_times),
        'delay_rate': kpi.flight_delays_count / len(kpi.flight_turnaround_times) if kpi.flight_turnaround_times else 0,
        'avg_delay_time': np.mean(kpi.flight_delay_minutes) if kpi.flight_delay_minutes else 0,
        'total_energy': kpi.total_energy_consumed,
        'total_charged': kpi.total_charge_kwh,
        'charge_events': kpi.total_charge_events,
        'charger1_util': kpi.charger_time_log['charging_1']['BUSY'] / sim_duration * 100,
        'charger2_util': kpi.charger_time_log['charging_2']['BUSY'] / sim_duration * 100,
        'epsilon': agent.epsilon,
    }

    return metrics, kpi, fleets, sim_duration


def plot_training_curves(history: list, output_dir: str):
    """Plot training curves."""
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Average Turnaround Time
    axes[0, 0].plot(df['episode'], df['avg_turnaround'], 'b-', alpha=0.3)
    axes[0, 0].plot(df['episode'], df['avg_turnaround'].rolling(10).mean(), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Avg Turnaround (min)')
    axes[0, 0].set_title('Average Turnaround Time')
    axes[0, 0].grid(True)

    # 2. Delay Rate
    axes[0, 1].plot(df['episode'], df['delay_rate'] * 100, 'r-', alpha=0.3)
    axes[0, 1].plot(df['episode'], (df['delay_rate'] * 100).rolling(10).mean(), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Delay Rate (%)')
    axes[0, 1].set_title('Flight Delay Rate')
    axes[0, 1].grid(True)

    # 3. Energy Consumption
    axes[0, 2].plot(df['episode'], df['total_energy'], 'g-', alpha=0.3)
    axes[0, 2].plot(df['episode'], df['total_energy'].rolling(10).mean(), 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Energy (kWh)')
    axes[0, 2].set_title('Total Energy Consumed')
    axes[0, 2].grid(True)

    # 4. Charge Events
    axes[1, 0].plot(df['episode'], df['charge_events'], 'm-', alpha=0.3)
    axes[1, 0].plot(df['episode'], df['charge_events'].rolling(10).mean(), 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Charge Events')
    axes[1, 0].set_title('Number of Charge Events')
    axes[1, 0].grid(True)

    # 5. Charger Utilization
    axes[1, 1].plot(df['episode'], df['charger1_util'], 'c-', alpha=0.5, label='Charger 1')
    axes[1, 1].plot(df['episode'], df['charger2_util'], 'y-', alpha=0.5, label='Charger 2')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Utilization (%)')
    axes[1, 1].set_title('Charger Utilization')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 6. Epsilon Decay
    axes[1, 2].plot(df['episode'], df['epsilon'], 'k-')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].set_title('Exploration Rate (Epsilon)')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=200)
    plt.close()

    print(f"[VIZ] Saved training curves to {output_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Train RL Agent for AGV Charging')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate every N episodes')
    parser.add_argument('--save_interval', type=int, default=25, help='Save model every N episodes')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("RL TRAINING - Airport eGSE Charging Agent")
    print(f"Episodes: {args.episodes}")
    print(f"Dispatching Rule: {cfg.DISPATCHING_RULE}")
    print(f"Fleet Size: {cfg.TOTAL_AMR_FLEET_SIZE}")
    print(f"{'=' * 60}\n")

    # Setup output directory
    output_dir = _setup_output_dir(f"RL_train_ep{args.episodes}")
    print(f"[INFO] Results will be saved to: {output_dir}")

    # Ensure RL is enabled
    cfg.USE_RL_CHARGING = True

    # Reset agent for fresh training
    reset_agent()

    # Load flight data
    df_flights, t_last_arrival = parse_arrivals(cfg.ARRIVAL_CSV, cfg.NUM_FLIGHTS)
    print(f"[INFO] Loaded {len(df_flights)} flights")
    print(f"[INFO] Time window: 0.0 to {t_last_arrival:.1f} min\n")

    # Training loop
    history = []
    best_delay_rate = float('inf')

    for episode in range(1, args.episodes + 1):
        print(f"Episode {episode}/{args.episodes}...", end=" ")

        metrics, kpi, fleets, sim_duration = run_episode(
            episode, df_flights, t_last_arrival, train_mode=True
        )
        history.append(metrics)

        print(f" | Turnaround: {metrics['avg_turnaround']:.1f}min, "
              f"Delays: {metrics['delayed_flights']}/{metrics['total_delays']} "
              f"({metrics['delay_rate']*100:.1f}%), "
              f"Eps: {metrics['epsilon']:.3f}")

        # Track best model
        if metrics['delay_rate'] < best_delay_rate:
            best_delay_rate = metrics['delay_rate']
            print(f"  -> New best delay rate: {best_delay_rate*100:.1f}%")

        # Periodic evaluation
        if episode % args.eval_interval == 0:
            print(f"\n--- Evaluation at Episode {episode} ---")
            eval_metrics, _, _, _ = run_episode(
                episode, df_flights, t_last_arrival, train_mode=False
            )
            print(f"  Eval Turnaround: {eval_metrics['avg_turnaround']:.1f}min")
            print(f"  Eval Delay Rate: {eval_metrics['delay_rate']*100:.1f}%")
            print(f"  Eval Energy: {eval_metrics['total_energy']:.1f} kWh\n")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Plot training curves
    plot_training_curves(history, output_dir)

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")

    final_metrics, final_kpi, final_fleets, final_duration = run_episode(
        args.episodes + 1, df_flights, t_last_arrival, train_mode=False
    )

    # Report summary
    final_kpi.report_summary(final_fleets, final_duration)

    # Export logs
    _export_logs(final_kpi, final_fleets, final_duration)
    _plot_gate_gantt()

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE. Results saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
