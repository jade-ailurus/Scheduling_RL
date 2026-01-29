"""
dr_utils.py

Domain randomization helpers for Scheduling_RL.

We treat the simulator as a black-box. Domain randomization is applied by
temporarily perturbing values in the `config` module and (optionally)
perturbing the arrivals dataframe before running an episode.

This is intentionally simple and "opt-in" per episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import random
import numpy as np


@dataclass
class PerturbationSpec:
    """
    Weak vs strong perturbation can be expressed by different ranges/std.

    battery_scale: multiplicative factor applied to DEFAULT_BATTERY_CAP_KWH
    charge_power_scale: multiplicative factor applied to CHARGE_POWER_KW
    travel_consume_scale: multiplicative factor applied to TRAVEL_CONSUME_POWER_KW
    service_time_noise_std: gaussian std (fractional) applied to SERVICE_TIMES values
    n_active_gates_range: if not None, randomly reassign each flight gate to a subset
                          of gates with size in this range (inclusive)
    """
    battery_scale: Tuple[float, float] = (1.0, 1.0)
    charge_power_scale: Tuple[float, float] = (1.0, 1.0)
    travel_consume_scale: Tuple[float, float] = (1.0, 1.0)
    service_time_noise_std: float = 0.0
    n_active_gates_range: Optional[Tuple[int, int]] = None


BASIC = PerturbationSpec(
    battery_scale=(1.0, 1.0),
    charge_power_scale=(1.0, 1.0),
    travel_consume_scale=(1.0, 1.0),
    service_time_noise_std=0.0,
    n_active_gates_range=None,
)

WEAK = PerturbationSpec(
    battery_scale=(0.85, 1.0),
    charge_power_scale=(0.9, 1.1),
    travel_consume_scale=(0.9, 1.1),
    service_time_noise_std=0.05,
    n_active_gates_range=(8, 10),
)

STRONG = PerturbationSpec(
    battery_scale=(0.60, 1.0),
    charge_power_scale=(0.75, 1.25),
    travel_consume_scale=(0.75, 1.25),
    service_time_noise_std=0.20,
    n_active_gates_range=(6, 10),
)


class DomainRandomizer:
    """
    Context manager that perturbs `config` module attributes for one episode,
    and restores them after exit.
    """

    def __init__(self, cfg_module: Any, spec: PerturbationSpec, rng: random.Random):
        self.cfg = cfg_module
        self.spec = spec
        self.rng = rng
        self._backup: Dict[str, Any] = {}

    def __enter__(self) -> "DomainRandomizer":
        self._backup = {
            "DEFAULT_BATTERY_CAP_KWH": getattr(self.cfg, "DEFAULT_BATTERY_CAP_KWH"),
            "CHARGE_POWER_KW": getattr(self.cfg, "CHARGE_POWER_KW"),
            "TRAVEL_CONSUME_POWER_KW": getattr(self.cfg, "TRAVEL_CONSUME_POWER_KW"),
            "SERVICE_TIMES": dict(getattr(self.cfg, "SERVICE_TIMES")),
            "AMR_SPEED_KPH": getattr(self.cfg, "AMR_SPEED_KPH"),
            "AMR_SPEED_DIST_PER_MIN": getattr(self.cfg, "AMR_SPEED_DIST_PER_MIN"),
        }

        # --- scalar multipliers ---
        bat_scale = self.rng.uniform(*self.spec.battery_scale)
        cp_scale = self.rng.uniform(*self.spec.charge_power_scale)
        tc_scale = self.rng.uniform(*self.spec.travel_consume_scale)

        setattr(self.cfg, "DEFAULT_BATTERY_CAP_KWH", self._backup["DEFAULT_BATTERY_CAP_KWH"] * bat_scale)
        setattr(self.cfg, "CHARGE_POWER_KW", self._backup["CHARGE_POWER_KW"] * cp_scale)
        setattr(self.cfg, "TRAVEL_CONSUME_POWER_KW", self._backup["TRAVEL_CONSUME_POWER_KW"] * tc_scale)

        # --- service times ---
        st_noise = float(self.spec.service_time_noise_std)
        if st_noise > 0.0:
            new_service_times = {}
            for k, v in self._backup["SERVICE_TIMES"].items():
                # multiplicative gaussian noise; clip to keep positive
                mult = 1.0 + self.rng.gauss(0.0, st_noise)
                mult = max(0.10, mult)
                new_service_times[k] = float(v) * mult
            setattr(self.cfg, "SERVICE_TIMES", new_service_times)

        # If speed changes are ever added, remember to recompute dist/min:
        # (we keep speed fixed by default here)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Restore all backed-up attributes
        for k, v in self._backup.items():
            setattr(self.cfg, k, v)


def perturb_arrivals_gates(df, gate_labels, spec: PerturbationSpec, rng: random.Random):
    """
    Optional perturbation: reassign each flight's gate to a random gate among
    a random subset of active gates.

    This is a crude way to induce capacity perturbations without rewriting the simulator.
    """
    if spec.n_active_gates_range is None:
        return df

    import pandas as pd  # local import

    lo, hi = spec.n_active_gates_range
    n_active = rng.randint(lo, hi)
    n_active = max(1, min(n_active, len(gate_labels)))

    active = rng.sample(list(gate_labels), n_active)
    df2 = df.copy()
    df2["GATE"] = [rng.choice(active) for _ in range(len(df2))]
    return df2
