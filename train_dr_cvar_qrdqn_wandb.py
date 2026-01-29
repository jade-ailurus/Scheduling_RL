"""
train_dr_cvar_qrdqn_wandb.py

Robust RL runner: Domain Randomization + QR-DQN (distributional) + CVaR policy.

Example:
    python train_dr_cvar_qrdqn_wandb.py --episodes 2000 --wandb --train-perturb strong --cvar-alpha 0.2

Notes
-----
- CVaR action selection uses the lower-tail of the return distribution
  (risk-averse). It is applied ONLY at action selection time.
- Training uses QR-DQN quantile regression loss.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import simpy

import config as cfg
import routing
import reporting
import sim_model_RL as sim_model

from agents import QRDQNAgent
from state_utils import extract_state
from wandb_utils import maybe_init_wandb, maybe_log, maybe_finish
from dr_utils import BASIC, WEAK, STRONG, DomainRandomizer, perturb_arrivals_gates, PerturbationSpec


def load_arrivals(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ARR_TIME" not in df.columns or "GATE" not in df.columns:
        raise ValueError("Expected columns: ARR_TIME, GATE")
    df["ARR_TIME"] = pd.to_datetime(df["ARR_TIME"])
    df["GATE"] = df["GATE"].apply(routing._norm_label)
    df = df[df["GATE"].isin(routing.GATE_LABELS)].copy()
    df.sort_values("ARR_TIME", inplace=True)
    df.reset_index(drop=True, inplace=True)

    t0 = df["ARR_TIME"].iloc[0]
    df["ARR_MIN"] = (df["ARR_TIME"] - t0).dt.total_seconds() / 60.0
    df["flight_id"] = np.arange(len(df), dtype=int)
    return df


def build_sim(env: simpy.Environment, df: pd.DataFrame) -> Tuple[Dict[str, sim_model.AMRFleet], Dict[str, simpy.Resource], reporting.KPIs, float]:
    cfg.USE_RL_CHARGING = True
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


def run_episode(
    df: pd.DataFrame,
    agent: QRDQNAgent,
    epsilon: float,
    train: bool,
    seed: int,
    dr_spec: PerturbationSpec,
    risk_mode: str = "cvar",
    cvar_alpha: float = 0.2,
) -> Dict[str, float]:
    import random
    rng = random.Random(seed)

    with DomainRandomizer(cfg, dr_spec, rng=rng):
        df_ep = perturb_arrivals_gates(df, routing.GATE_LABELS, dr_spec, rng=rng)

        env = simpy.Environment()
        fleets, gates, kpi, sim_duration = build_sim(env, df_ep)

        pending = {}  # (kind, unit_id) -> (s_norm, a, r)
        loss_vals = []
        decisions = 0

        def _drain_decisions() -> None:
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
                    s_raw = sv.raw
                    s = sv.normalized
                    key = (kind, unit_id)

                    # finalize pending
                    if train and key in pending:
                        s_prev, a_prev, r_prev = pending.pop(key)
                        agent.add(s_prev, a_prev, r_prev, s, done=0.0)
                        l = agent.learn()
                        if l is not None:
                            loss_vals.append(l)

                    # choose action (CVaR policy)
                    a = agent.act(s, epsilon=epsilon, risk_mode=risk_mode, cvar_alpha=cvar_alpha)

                    r = float(fleet._calc_charge_reward(s_raw, a, next_state_vec=None))

                    if train:
                        pending[key] = (s, a, r)

                    action_event.succeed(a)
                    decisions += 1

        try:
            while env.now < sim_duration:
                _drain_decisions()
                env.step()
            _drain_decisions()
        except simpy.core.EmptySchedule:
            pass

        if train:
            for (kind, unit_id), (s_prev, a_prev, r_prev) in pending.items():
                agent.add(s_prev, a_prev, r_prev, s_prev, done=1.0)
                l = agent.learn()
                if l is not None:
                    loss_vals.append(l)

        total_delay = float(sum(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0
        avg_delay = float(np.mean(kpi.flight_delay_minutes)) if kpi.flight_delay_minutes else 0.0
        delayed_flights = float(kpi.flight_delays_count)
        total_flights = float(len(kpi.flight_turnaround_times))

        return {
            "total_delay_min": total_delay,
            "avg_delay_min": avg_delay,
            "delayed_flights": delayed_flights,
            "total_flights": total_flights,
            "decision_points": float(decisions),
            "mean_quantile_loss": float(np.mean(loss_vals)) if loss_vals else 0.0,
        }


def evaluate(
    df: pd.DataFrame,
    agent: QRDQNAgent,
    seeds: list[int],
    dr_spec: PerturbationSpec,
    risk_mode: str,
    cvar_alpha: float,
) -> Dict[str, float]:
    all_m = []
    for s in seeds:
        m = run_episode(
            df,
            agent,
            epsilon=0.0,
            train=False,
            seed=s,
            dr_spec=dr_spec,
            risk_mode=risk_mode,
            cvar_alpha=cvar_alpha,
        )
        all_m.append(m)
    out = {}
    for k in all_m[0].keys():
        out[f"eval/{k}"] = float(np.mean([m[k] for m in all_m]))
        out[f"eval_std/{k}"] = float(np.std([m[k] for m in all_m]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=os.path.join("Data", "x-SFO-y_gate.csv"))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-path", type=str, default="checkpoints/dr_cvar_qrdqn.pt")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=int, default=1500)
    p.add_argument("--n-quantiles", type=int, default=51)
    p.add_argument("--risk-mode", type=str, default="cvar", choices=["mean", "cvar"])
    p.add_argument("--cvar-alpha", type=float, default=0.2)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="Scheduling_RL")
    p.add_argument("--wandb-group", type=str, default="dr_cvar_qrdqn")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--train-perturb", type=str, default="strong", choices=["basic", "weak", "strong"])
    p.add_argument("--eval-perturb", type=str, default="strong", choices=["basic", "weak", "strong"])
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    df = load_arrivals(args.data)

    train_spec = {"basic": BASIC, "weak": WEAK, "strong": STRONG}[args.train_perturb]
    eval_spec = {"basic": BASIC, "weak": WEAK, "strong": STRONG}[args.eval_perturb]

    agent = QRDQNAgent(state_dim=7, action_dim=3, n_quantiles=args.n_quantiles, seed=args.seed)

    run_name = args.run_name or f"qrdqn_s{args.seed}_{int(time.time())}"
    run = maybe_init_wandb(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        config={
            "algo": "DR-CVaR-QRDQN",
            "episodes": args.episodes,
            "seed": args.seed,
            "train_perturb": args.train_perturb,
            "eval_perturb": args.eval_perturb,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay_episodes": args.epsilon_decay_episodes,
            "n_quantiles": args.n_quantiles,
            "risk_mode": args.risk_mode,
            "cvar_alpha": args.cvar_alpha,
            "data": args.data,
        },
        enable=args.wandb,
    )


    # Baseline eval (untrained) so you can see "how many episodes until improvement"
    seeds0 = [args.seed + 9_000 + i for i in range(args.eval_episodes)]
    em0 = evaluate(df, agent, seeds=seeds0, dr_spec=eval_spec, risk_mode=args.risk_mode, cvar_alpha=args.cvar_alpha)
    maybe_log(run, {**em0, "baseline/episode": 0}, step=0)

    best = float("inf")
    best_ep = -1

    for ep in range(1, args.episodes + 1):
        frac = min(1.0, ep / float(max(1, args.epsilon_decay_episodes)))
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        m = run_episode(
            df,
            agent,
            epsilon=epsilon,
            train=True,
            seed=args.seed + ep,
            dr_spec=train_spec,
            risk_mode=args.risk_mode,
            cvar_alpha=args.cvar_alpha,
        )
        log = {f"train/{k}": v for k, v in m.items()}
        log["train/epsilon"] = float(epsilon)
        maybe_log(run, log, step=ep)

        if ep % args.eval_every == 0:
            seeds = [args.seed + 10_000 + i for i in range(args.eval_episodes)]
            em = evaluate(df, agent, seeds=seeds, dr_spec=eval_spec, risk_mode=args.risk_mode, cvar_alpha=args.cvar_alpha)
            maybe_log(run, em, step=ep)

            if em["eval/total_delay_min"] < best:
                best = em["eval/total_delay_min"]
                best_ep = ep
                agent.save(args.save_path)
                maybe_log(run, {"best/total_delay_min": best, "best/episode": best_ep}, step=ep)

    maybe_finish(run)
    print(f"[done] best eval total_delay_min={best:.2f} at episode {best_ep}; saved={args.save_path}")


if __name__ == "__main__":
    main()
