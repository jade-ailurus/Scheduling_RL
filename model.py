# ==============================================================================
# ===== AMR 클래스와 SimPy 프로세스 =====
# ==============================================================================

import simpy
import random
import statistics as stats

# config와 routing에서 필요한 모든 변수와 함수를 가져옵니다.
import config as cfg
from routing import (
    _get_path_waypoints, _calculate_path_distance_and_time,
    DEPOT_LABEL, CHARGER_LABELS
)
# reporting에서 LOG와 update_state 함수를 가져옵니다.
from reporting import LOG, update_state


class AMRUnit:
    """Represents a single AMR unit."""
    def __init__(self, unit_id: int, kind: str, capacity_kwh: float):
        self.unit_id = unit_id
        self.kind = kind
        self.global_id = f"{kind}_{unit_id}"

        self.soc_kwh = capacity_kwh
        self.capacity_kwh = capacity_kwh
        self.location = DEPOT_LABEL
        
        self.total_work_time = 0.0
        self.num_tasks = 0
        
        self.time_tracker = {"t_last_update": 0.0, "state": "IDLE"}

    @property
    def soc_percent(self) -> float:
        return self.soc_kwh / self.capacity_kwh

    def _update_time_kpi(self, env: simpy.Environment, new_state: str, kpi):
        """Updates the time-tracking KPI log."""
        t_now = env.now
        duration = t_now - self.time_tracker["t_last_update"]
        if duration > 0:
            last_state = self.time_tracker["state"]
            kpi.amr_time_log[(self.kind, self.unit_id)][last_state] += duration
        
        self.time_tracker["state"] = new_state
        self.time_tracker["t_last_update"] = t_now

    def consume_energy(self, duration_min: float, power_kw: float, kpi):
        """Consumes energy and updates SOC."""
        used_kwh = power_kw * (duration_min / 60.0)
        self.soc_kwh = max(0.0, self.soc_kwh - used_kwh)
        kpi.total_energy_consumed += used_kwh

class ChargerBank:
    """Represents a charging station with configurable capacity."""
    def __init__(self, env: simpy.Environment, name: str, kpi, capacity: int):
        self.env = env
        self.name = name
        self.res = simpy.Resource(env, capacity=capacity)
        self.kpi = kpi
        self.time_tracker = {"t_last_update": 0.0, "queue": 0}
        
    def _update_util_kpi(self):
        """Updates charger utilization KPI."""
        t_now = self.env.now
        queue = len(self.res.queue)
        
        if self.time_tracker["queue"] > 0:
             duration = t_now - self.time_tracker["t_last_update"]
             self.kpi.charger_time_log[self.name]["BUSY"] += duration
        
        self.time_tracker["t_last_update"] = t_now
        self.time_tracker["queue"] = queue

class AMRFleet:
    """Manages a fleet of AMRUnits of a specific kind (e.g., 'GPU')."""
    def __init__(self, env: simpy.Environment, kind: str, size: int,
                 chargers: dict, kpi):
        self.env = env
        self.kind = kind

        if kind == 'GPU':
            capacity = cfg.GPU_CONFIG['BATTERY_CAP_KWH']
        else:
            capacity = cfg.DEFAULT_BATTERY_CAP_KWH

        self.units = [AMRUnit(i, kind, capacity_kwh=capacity) for i in range(size)]
        
        self.available = set(u.global_id for u in self.units)
        self.unit_map = {u.global_id: u for u in self.units}
        
        self.lock = simpy.Resource(env, capacity=1)
        self.chargers = chargers
        self.kpi = kpi

    def _travel(self, unit: AMRUnit, dst: str, flight_id: str = None):
        """Generator for AMR travel using (x, y) router."""
        if unit.location == dst:
            yield self.env.timeout(0)
            return

        unit._update_time_kpi(self.env, "TRAVEL", self.kpi)
        t_start = self.env.now
        src = unit.location
        
        try:
            waypoints = _get_path_waypoints(src, dst)
        except NotImplementedError:
            print(f"[ERROR] No path defined: {src} -> {dst}. Halting.")
            yield self.env.timeout(99999)
            return
            
        distance, travel_time = _calculate_path_distance_and_time(waypoints)
        
        if travel_time > 0:
            unit.consume_energy(travel_time, cfg.TRAVEL_CONSUME_POWER_KW, self.kpi)
            yield self.env.timeout(travel_time)
        
        unit.location = dst
        unit.total_work_time += travel_time
        self.kpi.total_travel_distance += distance

        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "travel_end",
            t_start=t_start, src=src, dst=dst, flight_id=flight_id, 
            path=waypoints
        )

    def _service(self, unit: AMRUnit, duration_min: float, flight_id: str, task: str):
        """Generator for AMR service, consuming energy."""
        unit._update_time_kpi(self.env, "SERVICE", self.kpi)
        t_start = self.env.now

        unit.consume_energy(duration_min, cfg.DEFAULT_SERVICE_CONSUME_POWER_KW, self.kpi)
        yield self.env.timeout(duration_min)

        unit.total_work_time += duration_min
        unit.num_tasks += 1
        
        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "service_end",
            t_start=t_start, duration=duration_min, flight_id=flight_id, task=task
        )

    def _charge(self, unit: AMRUnit, charger_name: str):
        """Generator for charging at a specific charger."""
        charger = self.chargers[charger_name]
        
        yield self.env.process(self._travel(unit, charger_name))

        unit._update_time_kpi(self.env, "Q_CHARGE", self.kpi)
        t_queue_start = self.env.now
        charger._update_util_kpi()
        
        with charger.res.request() as creq:
            yield creq
            
            charger._update_util_kpi()
            t_charge_start = self.env.now
            unit._update_time_kpi(self.env, "CHARGING", self.kpi)
            LOG.log_amr(
                t_charge_start, unit.global_id, self.kind, "charge_start",
                charger=charger_name, queue_time=(t_charge_start - t_queue_start)
            )

            need_kwh = max(0.0, unit.capacity_kwh - unit.soc_kwh)
            if need_kwh > 0:
                hours_to_charge = need_kwh / cfg.CHARGE_POWER_KW
                duration_min = hours_to_charge * 60.0
                yield self.env.timeout(duration_min)
                
                unit.soc_kwh = unit.capacity_kwh
                self.kpi.total_charge_events += 1
                self.kpi.total_charge_kwh += need_kwh

        charger._update_util_kpi()
        LOG.log_amr(
            self.env.now, unit.global_id, self.kind, "charge_end",
            charger=charger_name
        )

    def _find_shortestQ_charger(self, unit: AMRUnit) -> (str, float):
        """Finds the BEST charger (shortest queue, then shortest time)."""
        options = []
        
        for name, charger_bank in self.chargers.items():
            try:
                queue_len = len(charger_bank.res.queue)
                waypoints = _get_path_waypoints(unit.location, name)
                _, time = _calculate_path_distance_and_time(waypoints)
                options.append((queue_len, time, name))
            except NotImplementedError:
                continue
        
        if not options:
            return None, float('inf')
            
        options.sort(key=lambda x: (x[0], x[1]))
        
        best_queue, best_time, best_name = options[0]
        return best_name, best_time
        
    def _get_eligible_units(self, required_kwh: float) -> list:
        """Finds units at DEPOT with enough charge."""
        eligible_units = []
        for unit_id in list(self.available):
            unit = self.unit_map[unit_id]
            if unit.location == DEPOT_LABEL and unit.soc_kwh >= required_kwh:
                eligible_units.append(unit)
        return eligible_units

    def _select_units_by_rule(self, eligible: list, n: int) -> list:
        """Applies the configured dispatching rule."""
        if len(eligible) < n:
            return []

        if cfg.DISPATCHING_RULE == 'FIFO':
            selected = eligible[:n]
        
        elif cfg.DISPATCHING_RULE == 'RANDOM':
            selected = random.sample(eligible, n)
            
        elif cfg.DISPATCHING_RULE == 'LEAST_UTILIZED':
            eligible.sort(key=lambda u: (u.total_work_time, u.num_tasks))
            selected = eligible[:n]

        elif cfg.DISPATCHING_RULE == 'BIDDING':
            utils = [u.total_work_time for u in eligible]
            max_util = max(utils) if utils else 1.0
            
            def calculate_bid(unit):
                soc_score = unit.soc_percent
                util_score = 1.0 - (unit.total_work_time / max_util) if max_util > 0 else 1.0
                bid = (soc_score * 0.7) + (util_score * 0.3)
                return bid

            eligible.sort(key=lambda u: calculate_bid(u), reverse=True)
            selected = eligible[:n]

        else:
            raise ValueError(f"Unknown DISPATCHING_RULE: {cfg.DISPATCHING_RULE}")

        return selected

    def request_units(self, n: int, required_kwh: float, task: str):
        """Generator to request 'n' available units based on rules."""
        
        def _get_units():
            while True:
                with self.lock.request() as req:
                    yield req
                    
                    eligible = self._get_eligible_units(required_kwh)
                    selected_units = self._select_units_by_rule(eligible, n)
                    
                    if len(selected_units) == n:
                        for unit in selected_units:
                            self.available.remove(unit.global_id)
                            unit._update_time_kpi(self.env, "Q_TASK", self.kpi)
                        return selected_units
                
                yield self.env.timeout(0.1)

        t_request = self.env.now
        selected = yield self.env.process(_get_units())
        
        wait_time = self.env.now - t_request
        LOG.log_amr(
            self.env.now, f"Fleet_{self.kind}", self.kind, "dispatch_success",
            task=task, wait_time=wait_time, n=n
        )
        return selected

    def release_units(self, units: list, task_end_location: str, all_fleets: dict):
        """Generator to release units, triggering charge/return logic."""

        def _unit_return_logic(unit: AMRUnit, all_fleEts: dict):
            update_state(self.env, "amr_task_end", self.kpi, all_fleets)
            unit.location = task_end_location
            
            if unit.soc_percent < cfg.CHARGE_TRIGGER_SOC:
                charger_name, _ = self._find_shortestQ_charger(unit)
                if charger_name:
                    yield self.env.process(self._charge(unit, charger_name))
                else:
                    print(f"[WARN] {unit.global_id} needs charge but no path found.")
            
            if unit.location != DEPOT_LABEL:
                yield self.env.process(self._travel(unit, DEPOT_LABEL))
            
            unit._update_time_kpi(self.env, "IDLE", self.kpi)
            with self.lock.request() as req:
                yield req
                self.available.add(unit.global_id)

        for u in units:
            self.env.process(_unit_return_logic(u, all_fleets))
        yield self.env.timeout(0)

# ==============================================================================
# ===== SIMULATION PROCESS LOGIC =====
# ==============================================================================

def _task_process(
    env: simpy.Environment, 
    flight_id: str, 
    gate_label: str,
    task_name: str, 
    fleets: dict,
    kpi,
    gpu_arrived_event: simpy.Event,
    baggage_out_done_event: simpy.Event = None
):
    """Core generator for a single eGSE task (e.g., 'FUEL')."""
    fleet_name = cfg.TASK_TO_FLEET_MAP[task_name]
    fleet = fleets[fleet_name]
    num_units = cfg.REQUIRED_UNITS[task_name]
    service_duration = cfg.SERVICE_TIMES[task_name]

    # 1. Calculate required energy
    try:
        _, t_to_gate = _calculate_path_distance_and_time(_get_path_waypoints(DEPOT_LABEL, gate_label))
        _, t_to_depot = _calculate_path_distance_and_time(_get_path_waypoints(gate_label, DEPOT_LABEL))
    except NotImplementedError:
        print(f"[ERROR] Cannot calculate energy for {task_name} at {gate_label}.")
        return
        
    travel_time = t_to_gate + t_to_depot
    required_kwh = (travel_time * cfg.TRAVEL_CONSUME_POWER_KW + 
                    service_duration * cfg.DEFAULT_SERVICE_CONSUME_POWER_KW) / 60.0
    
    # 2. Request units
    units = yield env.process(fleet.request_units(num_units, required_kwh, task_name))
    unit = units[0]
    LOG.log_amr(
        env.now, unit.global_id, fleet_name, "dispatch_assigned", 
        flight_id=flight_id, task=task_name
    )
    
    # 3. Travel to gate
    yield env.process(fleet._travel(unit, gate_label, flight_id=flight_id))
    
    # 4. === EVENT SYNCHRONIZATION ===
    unit._update_time_kpi(env, "Q_TASK", kpi) 
    t_wait_start = env.now

    # Wait for GPU
    yield gpu_arrived_event
    wait_time = env.now - t_wait_start
    kpi.gpu_arrival_wait_times.append(wait_time)
    
    LOG.log_amr(
        env.now, unit.global_id, fleet_name, "gpu_wait_over", 
        flight_id=flight_id, wait_time=wait_time
    )

    if task_name == 'BAGGAGE_IN':
        # Wait for BAGGAGE_OUT
        t_bag_wait_start = env.now
        yield baggage_out_done_event
        wait_time_bag = env.now - t_bag_wait_start
        kpi.baggage_in_wait_times.append(wait_time_bag)
        unit._update_time_kpi(env, "Q_TASK", kpi) # Log time waiting for BagOut
        LOG.log_amr(
            env.now, unit.global_id, fleet_name, "bag_out_wait_over", 
            flight_id=flight_id, wait_time=wait_time_bag
        )

    # 5. Perform Service
    if service_duration > 0:
        yield env.process(fleet._service(unit, service_duration, flight_id, task_name))
        
    # 6. Signal completion
    if task_name == 'BAGGAGE_OUT':
        baggage_out_done_event.succeed()
        LOG.log_amr(
            env.now, unit.global_id, fleet_name, "bag_out_done_signal", 
            flight_id=flight_id
        )

    LOG.log_flight(env.now, flight_id, "task_completed", task=task_name)

    # 7. Release unit
    env.process(fleet.release_units(units, gate_label, fleets))

def flight_process(env: simpy.Environment, flight_id: str, gate_label: str, 
                   gates: dict, fleets: dict, kpi):
    """Main generator for a single flight."""
    
    # 1. Wait for gate
    t_arrival = env.now
    LOG.log_flight(t_arrival, flight_id, "gate_queued", gate=gate_label)
    
    with gates[gate_label].request() as greq:
        yield greq
        
        t_gate_start = env.now
        gate_wait = t_gate_start - t_arrival
        kpi.gate_wait_times.append(gate_wait)
        LOG.log_flight(t_gate_start, flight_id, "gate_start", gate=gate_label, wait_time=gate_wait)

        update_state(env, "flight_arrival", kpi, fleets)

        # 2. Create event triggers
        gpu_arrived = env.event()
        baggage_out_done = env.event()

        # 3. Handle GPU and OTHER tasks in parallel
        gpu_fleet = fleets['GPU']

        # 3a. Define GPU sub-process
        def _gpu_main_task_process(env, flight_id, gate_label, kpi_inst, t_gate_arrival, gpu_arrived_event, fleet_to_use):
            """Process for the GPU's entire lifecycle at the gate."""
            
            # 1. Request GPU Unit
            try:
                _, t_to_gate = _calculate_path_distance_and_time(_get_path_waypoints(DEPOT_LABEL, gate_label))
            except NotImplementedError:
                 t_to_gate = 1.0
            required_kwh = (t_to_gate * cfg.TRAVEL_CONSUME_POWER_KW) / 60.0
            
            units = yield env.process(fleet_to_use.request_units(1, required_kwh, 'GPU'))
            gpu_unit = units[0]

            # 2. Travel
            yield env.process(fleet_to_use._travel(gpu_unit, gate_label, flight_id=flight_id))
            
            # 3. Arrive & Signal
            gpu_arrived_event.succeed()
            LOG.log_amr(env.now, gpu_unit.global_id, 'GPU', "gpu_arrived_signal", flight_id=flight_id)
            
            # 4. Return unit AND service start time
            t_service_start = env.now
            gpu_unit._update_time_kpi(env, "SERVICE", kpi_inst)
            return (gpu_unit, t_service_start)
        
        # 3b. Start all tasks in parallel
        tasks_to_start = [
            'FUEL', 'WATER', 'CLEAN', 'CATERING', 'BAGGAGE_OUT', 'BAGGAGE_IN'
        ]
        
        gpu_process = env.process(
            _gpu_main_task_process(env, flight_id, gate_label, kpi, t_gate_start, gpu_arrived, gpu_fleet)
        )
        
        all_other_task_processes = []
        for task in tasks_to_start:
            proc = env.process(
                _task_process(
                    env, flight_id, gate_label, task, fleets, kpi,
                    gpu_arrived_event=gpu_arrived,
                    baggage_out_done_event=baggage_out_done
                )
            )
            all_other_task_processes.append(proc)
        
        # 4. Wait for ALL OTHER tasks to complete
        yield simpy.AllOf(env, all_other_task_processes)
        
        # 5. Get GPU unit and consume service energy
        (gpu_unit, t_service_start) = yield gpu_process
        t_service_end = env.now
        
        gpu_service_duration = t_service_end - t_service_start
        if gpu_service_duration > 0:
            power_kw = cfg.GPU_CONFIG['SERVICE_CONSUME_POWER_KW']
            gpu_unit.consume_energy(gpu_service_duration, power_kw, kpi)
        
        LOG.log_flight(env.now, flight_id, "task_completed", task='GPU')
        env.process(gpu_fleet.release_units([gpu_unit], gate_label, fleets))

        # 6. All services done, release gate
        t_gate_end = env.now
        turnaround_time = t_gate_end - t_gate_start
        kpi.flight_turnaround_times.append(turnaround_time)
        
        if turnaround_time > cfg.TARGET_TURNAROUND_MINUTES:
            kpi.flight_delays_count += 1
            kpi.flight_delay_minutes.append(turnaround_time - cfg.TARGET_TURNAROUND_MINUTES)

        LOG.log_flight(t_gate_end, flight_id, "gate_end", gate=gate_label, duration=turnaround_time)

def flight_starter(env, t_start_min: float, *args):
    """Schedules a flight_process to start at a specific time."""
    if t_start_min > env.now:
        yield env.timeout(t_start_min - env.now)
    env.process(flight_process(env, *args))