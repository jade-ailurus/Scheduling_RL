"""
benchmark_compare_wandb.py

Compare:
- Heuristic baselines (dispatching rules with heuristic charging)
- Trained DQN charging
- Trained DR-CVaR-QRDQN charging

Across perturbation regimes: basic / weak / strong.

This script produces:
- console summary
- optional W&B table logging
- a CSV summary in ./benchmark_out/

Example:
    python benchmark_compare_wandb.py --wandb \\
        --dqn-ckpt checkpoints/dqn_charging.pt \\
        --qrdqn-ckpt checkpoints/dr_cvar_qrdqn.pt \\
        --n-seeds 20
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

import config as cfg
import routing
import reporting
import sim_model_RL as sim_model

from agents import DQNAgent, QRDQNAgent
from state_utils import extract_state
from wandb_utils import maybe_init_wandb, maybe_log, maybe_finish
from dr_utils import BASIC, WEAK, STRONG, DomainRandomizer, perturb_arrivals_gates, PerturbationSpec


def load_arrivals(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ARR_TIME"] = pd.to_datetime(df["ARR_TIME"])
    df["GATE"] = df["GATE"].apply(routing._norm_label)
    df = df[df["GATE"].isin(routing.GATE_LABELS)].copy()
    df.sort_values("ARR_TIME", inplace=True)
    df.reset_index(drop=True, inplace=True)

    t0 = df["ARR_TIME"].iloc[0]
    df["ARR_MIN"] = (df["ARR_TIME"] - t0).dt.total_seconds() / 60.0
    df["flight_id"] = np.arange(len(df), dtype=int)
    return df


def _build_common(env: simpy.Environment, df: pd.DataFrame) -> Tuple[Dict[str, sim_model.AMRFleet], Dict[str, simpy.Resource], reporting.KPIs, float]:
    kpi = reporting.KPIs(cfg.AMR_FLEET_CONFIG)
    chargers = sim_model.ChargerBank(env, cfg.CHARGER_CAPACITY, kpi)
    fleets = {
        kind: sim_model.AMRFleet(env, kind, size, chargers, kpi, rl_mode=True)
        for kind, size in cfg.AMR_FLEET_CONFIG.items()
    }
    gates = {g: simpy.Resource(env, capacity=1) for g in routing.GATE_LABELS}

    for _, row in df.iterrows():
        env.process(
            sim_model.flight_starter(
                env=env,
                gates=gates,
                fleets=fleets,
                arr_time=float(row["ARR_MIN"]),
                flight_id=int(row["flight_id"]),
                gate_id=str(row["GATE"]),
                kpi=kpi,
            )
        )

    sim_duration = float(df["ARR_MIN"].max()) + float(cfg.SIM_BUFFER_MIN)
    return fleets, gates, kpi, sim_duration


def run_episode_heuristic(df: pd.DataFrame, seed: int, dr_spec: PerturbationSpec, dispatch_rule: str) -> Dict[str, float]:
    import random
    rng = random.Random(seed)

    with DomainRandomizer(cfg, dr_spec, rng=rng):
        df_ep = perturb_arrivals_gates(df, routing.GATE_LABELS, dr_spec, rng=rng)

        cfg.USE_RL_CHARGING = False
        cfg.DISPATCHING_RULE = dispatch_rule

        env = simpy.Environment()
        fleets, gates, kpi, sim_duration = _build_common(env, df_ep)

        env.run(until=sim_duration)

        total_delay = float(sum(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0
        avg_delay = float(np.mean(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0

        return {
            "total_delay_min": total_delay,
            "avg_delay_min": avg_delay,
            "delayed_flights": float(kpi.flight_delays_count),
            "total_flights": float(len(kpi.flight_turnaround_times)),
        }


def run_episode_rl(
    df: pd.DataFrame,
    seed: int,
    dr_spec: PerturbationSpec,
    dispatch_rule: str,
    agent,
    algo: str,
    risk_mode: str = "mean",
    cvar_alpha: float = 0.2,
) -> Dict[str, float]:
    import random
    rng = random.Random(seed)

    with DomainRandomizer(cfg, dr_spec, rng=rng):
        df_ep = perturb_arrivals_gates(df, routing.GATE_LABELS, dr_spec, rng=rng)

        cfg.USE_RL_CHARGING = True
        cfg.DISPATCHING_RULE = dispatch_rule

        env = simpy.Environment()
        fleets, gates, kpi, sim_duration = _build_common(env, df_ep)

        decisions = 0

        def _drain() -> None:
            nonlocal decisions
            for kind, fleet in fleets.items():
                while fleet.decision_queue:
                    item = fleet.decision_queue.pop(0)
                    unit_id = int(item["unit_id"])
                    snapshot = item["snapshot"]
                    action_event = item["action_event"]

                    sv = extract_state(
                        snapshot=snapshot,
                        unit_id=unit_id,
                        env_now=float(env.now),
                        sim_duration_min=sim_duration,
                        cfg_module=cfg,
                        routing_module=routing,
                    )
                    s = sv.normalized

                    if algo == "DQN":
                        a = agent.act(s, epsilon=0.0)
                    elif algo == "QRDQN":
                        a = agent.act(s, epsilon=0.0, risk_mode=risk_mode, cvar_alpha=cvar_alpha)
                    else:
                        raise ValueError(algo)

                    action_event.succeed(a)
                    decisions += 1

        try:
            while env.now < sim_duration:
                _drain()
                env.step()
            _drain()
        except simpy.core.EmptySchedule:
            pass

        total_delay = float(sum(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0
        avg_delay = float(np.mean(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0

        return {
            "total_delay_min": total_delay,
            "avg_delay_min": avg_delay,
            "delayed_flights": float(kpi.flight_delays_count),
            "total_flights": float(len(kpi.flight_turnaround_times)),
            "decision_points": float(decisions),
        }


def agg(rows: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    out = {}
    keys = rows[0].keys()
    for k in keys:
        out[f"{prefix}/{k}_mean"] = float(np.mean([r[k] for r in rows]))
        out[f"{prefix}/{k}_std"] = float(np.std([r[k] for r in rows]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=os.path.join("Data", "x-SFO-y_gate.csv"))
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="Scheduling_RL")
    p.add_argument("--wandb-group", type=str, default="benchmarks")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--heuristic-dispatch", type=str, default="BIDDING", choices=["FIFO", "RANDOM", "LEAST_UTILIZED", "BIDDING"])
    p.add_argument("--rl-dispatch", type=str, default="FIFO", choices=["FIFO", "RANDOM", "LEAST_UTILIZED", "BIDDING"])
    p.add_argument("--dqn-ckpt", type=str, default="")
    p.add_argument("--qrdqn-ckpt", type=str, default="")
    p.add_argument("--qrdqn-quantiles", type=int, default=51)
    p.add_argument("--risk-mode", type=str, default="cvar", choices=["mean", "cvar"])
    p.add_argument("--cvar-alpha", type=float, default=0.2)
    args = p.parse_args()

    df = load_arrivals(args.data)

    run_name = args.run_name or f"bench_{int(time.time())}"
    run = maybe_init_wandb(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        config={
            "n_seeds": args.n_seeds,
            "seed0": args.seed0,
            "heuristic_dispatch": args.heuristic_dispatch,
            "rl_dispatch": args.rl_dispatch,
            "risk_mode": args.risk_mode,
            "cvar_alpha": args.cvar_alpha,
        },
        enable=args.wandb,
    )

    specs = {"basic": BASIC, "weak": WEAK, "strong": STRONG}

    results_table = []

    # 1) Heuristic baseline
    for spec_name, spec in specs.items():
        rows = [run_episode_heuristic(df, seed=args.seed0 + i, dr_spec=spec, dispatch_rule=args.heuristic_dispatch) for i in range(args.n_seeds)]
        m = agg(rows, prefix=f"heuristic_{args.heuristic_dispatch}_{spec_name}")
        maybe_log(run, m)
        results_table.append({"method": f"heuristic_{args.heuristic_dispatch}", "perturb": spec_name, **m})

    # 2) DQN
    if args.dqn_ckpt:
        dqn = DQNAgent(state_dim=7, action_dim=3, seed=args.seed0)
        dqn.load(args.dqn_ckpt)
        for spec_name, spec in specs.items():
            rows = [run_episode_rl(df, seed=args.seed0 + i, dr_spec=spec, dispatch_rule=args.rl_dispatch, agent=dqn, algo="DQN") for i in range(args.n_seeds)]
            m = agg(rows, prefix=f"dqn_{spec_name}")
            maybe_log(run, m)
            results_table.append({"method": "DQN", "perturb": spec_name, **m})

    # 3) DR-CVaR-QRDQN
    if args.qrdqn_ckpt:
        qrdqn = QRDQNAgent(state_dim=7, action_dim=3, n_quantiles=args.qrdqn_quantiles, seed=args.seed0)
        qrdqn.load(args.qrdqn_ckpt)
        for spec_name, spec in specs.items():
            rows = [
                run_episode_rl(
                    df,
                    seed=args.seed0 + i,
                    dr_spec=spec,
                    dispatch_rule=args.rl_dispatch,
                    agent=qrdqn,
                    algo="QRDQN",
                    risk_mode=args.risk_mode,
                    cvar_alpha=args.cvar_alpha,
                )
                for i in range(args.n_seeds)
            ]
            m = agg(rows, prefix=f"dr_cvar_qrdqn_{spec_name}")
            maybe_log(run, m)
            results_table.append({"method": "DR-CVaR-QRDQN", "perturb": spec_name, **m})

    # Save CSV
    os.makedirs("benchmark_out", exist_ok=True)
    out_path = os.path.join("benchmark_out", f"summary_{int(time.time())}.csv")
    flat_rows = []
    for r in results_table:
        # flatten nested keys: keep as-is
        flat_rows.append(r)
    pd.DataFrame(flat_rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")

    maybe_finish(run)


if __name__ == "__main__":
    main()
