#!/usr/bin/env python3
"""Live RL governor control of the real M/G/1 client-server workload.

Launches the unmodified MG1 Java simulation (client.LoadGenerator -> UDP ->
Server.Server, single worker on an isolated CPU) and, while it runs, a
sequential DDQN picks (ondemand sampling_rate, up_threshold, frequency bin)
every decision window.  Reward per window is EDP = target_power * MRT^2
relative to a rolling reference measured at the stock ondemand config
(10 000 us / 95 % / 2100 MHz ceiling), the exact baseline of the offline
dataset.

Response times are streamed live by the instrumented server through the
MG1_TELEMETRY_FILE environment variable (per-job CSV rows appended on job
completion).  Power comes from RAPL minus the other-cores idle attribution,
matching the MG1 paper's measured-core-power
convention.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cpu_control import (
    CpuFrequencyController,
    RaplDomain,
    cpu_stat,
    online_cpu_count,
    read_max_temperature_c,
    utilization_between,
)
from mg1_common import (
    CsvAppender,
    attach_state,
    calibrate_package_idle_power,
    edp_reward,
    training_episode_position,
)
from sequential_ddqn import (
    ACTION_GRID,
    DDQNConfig,
    SequentialDDQN,
    ensemble_decision,
    epsilon_by_step,
)
from config_only_ddqn import ConfigOnlyDDQN, ConfigOnlyWrapper

# The MG1 calc kernel performs 6 source-level FP ops per iteration:
# sin, cos, multiply, pow(i,i), divide, pow(...,i).
MG1_FLOPS_PER_ITERATION = 6

RUN_FIELDS = [
    "phase",
    "run",
    "episode_id",
    "episode_step",
    "timestamp",
    "config_id",
    "sampling_rate_us",
    "threshold_pct",
    "frequency_bin_id",
    "requested_frequency_khz",
    "mapped_frequency_khz",
    "window_seconds",
    "jobs_completed",
    "arrival_rate_jobs_per_s",
    "requested_lambda",
    "mean_response_time_s",
    "mean_service_time_s",
    "mean_wait_time_s",
    "mean_measured_frequency_khz",
    "utilization",
    "epsilon",
    "exploratory",
    "greedy_config_id",
    "greedy_frequency_bin_id",
    "chosen_q",
    "max_q",
    "rapl_domain",
    "domain_energy_j",
    "domain_power_w",
    "other_cores_idle_power_w",
    "estimated_target_power_w",
    "power_basis",
    "power_clamped",
    "temperature_c",
    "decision_trigger",
    "phase_baseline_utilization",
    "phase_recent_utilization",
    "edp_w_s2",
    "reference_edp_w_s2",
    "reference_baseline_source",
    "reference_baseline_samples",
    "mean_flops_per_job",
    "total_flops",
    "gflops_per_s",
    "gflops_per_w",
    "reward",
    "done",
    "replay_size",
    "training_loss",
    "training_updates",
    "training_td_error",
    "training_td_error_abs",
    "training_target_q_mean",
    "training_chosen_q_mean",
    "training_max_q_mean",
    "training_q_gap_mean",
    "training_greedy_rate",
    "state_0",
    "state_1",
    "state_2",
    "state_3",
    "state_4",
    "state_5",
    "state_6",
    "next_state_0",
    "next_state_1",
    "next_state_2",
    "next_state_3",
    "next_state_4",
    "next_state_5",
    "next_state_6",
    "telemetry_path",
    "server_pid",
    "client_pid",
]


LEARNING_FIELDS = [
    "step",
    "run",
    "episode_id",
    "episode_step",
    "loss",
    "td_error",
    "td_error_abs",
    "target_q",
    "chosen_q",
    "max_q",
    "q_gap",
    "greedy_rate",
    "replay_size",
    "epsilon",
    "exploratory",
    "reward",
    "utilization",
    "edp_w_s2",
]


class Mg1Proc:
    """Tracks the MG1 server + client Popen handles."""

    def __init__(self) -> None:
        self.server: Optional[subprocess.Popen] = None
        self.client: Optional[subprocess.Popen] = None

    def alive(self) -> bool:
        alive = True
        for handle in (self.server, self.client):
            if handle is not None:
                alive = alive and handle.poll() is None
        return alive

    def terminate(self) -> None:
        for handle in (self.server, self.client):
            if handle is not None and handle.poll() is None:
                try:
                    handle.terminate()
                except OSError:
                    pass
        for handle in (self.server, self.client):
            if handle is not None:
                try:
                    handle.wait(timeout=10.0)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        handle.kill()
                    except OSError:
                        pass


class TelemetryReader:
    """Tails the MG1_PER_JOB telemetry CSV, returning per-window aggregates.

    Each appended row is wall_ms,response_ns,calc_ns,wait_ns,cpu_ns,repeat.
    Successive ``read()`` calls yield only the rows written since the last
    call, so one call per decision window produces windowed response-time
    statistics for the single-server queue.
    """

    FIELDS = ["wall_ms", "response_ns", "calc_ns", "wait_ns", "cpu_ns", "repeat"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.offset = 0
        self.start_wall_ms: Optional[int] = None
        self._open_handle: Optional[object] = None

    def reset(self) -> None:
        if self.path.exists():
            self.path.write_bytes(b"")
        self.offset = 0
        self.start_wall_ms = None

    def clear_start(self) -> None:
        self.start_wall_ms = None

    def flush(self) -> None:
        """Advance past any pending rows (e.g. warmup data) without counting them."""
        if self.path.exists():
            self.offset = self.path.stat().st_size
        self.start_wall_ms = None

    def base_wall_ms(self) -> Optional[int]:
        """Wall-clock timestamp (ms) of the first job seen, for drift checks."""
        return self.start_wall_ms

    def read(self) -> Tuple[List[Dict[str, float]], int]:
        rows: List[Dict[str, float]] = []
        if not self.path.exists():
            return rows, self.offset
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) != len(self.FIELDS):
                    continue
                try:
                    values = [float(value) for value in parts]
                except ValueError:
                    continue
                row = dict(zip(self.FIELDS, values))
                rows.append(row)
                if self.start_wall_ms is None:
                    self.start_wall_ms = int(row["wall_ms"])
            self.offset = handle.tell()
        return rows, self.offset


def aggregate_rows(rows: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Summarise the telemetry rows collected in one decision window."""
    if not rows:
        return None
    total_response_ns = sum(float(row["response_ns"]) for row in rows)
    total_calc_ns = sum(float(row["calc_ns"]) for row in rows)
    total_wait_ns = sum(float(row["wait_ns"]) for row in rows)
    count = len(rows)
    total_flops = sum(float(row.get("repeat", 0)) * MG1_FLOPS_PER_ITERATION for row in rows)
    return {
        "jobs_completed": count,
        "mean_response_time_s": total_response_ns / count / 1e9,
        "mean_service_time_s": total_calc_ns / count / 1e9,
        "mean_wait_time_s": total_wait_ns / count / 1e9,
        "total_flops": total_flops,
        "mean_flops_per_job": total_flops / count if count else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live sequential DDQN governor control of the M/G/1 "
            "client-server simulation on real hardware"
        )
    )
    parser.add_argument("--target-cpu", type=int, required=True)
    parser.add_argument("--controller-cpu", type=int, required=True)
    parser.add_argument("--client-cpu", type=int, required=True)
    parser.add_argument("--mg1-root", type=Path, required=True)
    parser.add_argument("--lambda", type=float, dest="arrival_rate")
    parser.add_argument(
        "--lambda-schedule",
        type=str,
        default="",
        help="Comma-separated list of lambdas to cycle through during training "
        "(e.g., '0.8,1.6,2.4,3.2,4.0,4.8,5.6,6.4,7.2,7.5'). "
        "Enables universal-model mode. Implies --windows-per-lambda is set "
        "to maintain total training windows.",
    )
    parser.add_argument(
        "--windows-per-lambda",
        type=int,
        default=15,
        help="Number of training windows per lambda in schedule mode (default: 15). "
        "Total training windows = len(schedule) * windows_per_lambda.",
    )
    parser.add_argument(
        "--eval-lambdas",
        type=str,
        default="",
        help="Comma-separated lambdas for post-training evaluation. "
        "Defaults to lambda-schedule if not set.",
    )
    parser.add_argument("--client-repeat", type=int, default=1_000_000)
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results/mg1_rl"))
    parser.add_argument("--session-name", type=str, default="")
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--warmup-s", type=float, default=45.0)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument(
        "--decision-trigger",
        choices=("fixed", "phase"),
        default="fixed",
        help=(
            "'fixed' holds each chosen action for a fixed window of --window-s "
            "seconds; 'phase' re-decides when the worker-core CPU utilization "
            "drifts more than --phase-util-threshold-pct away from the phase "
            "baseline (a new action is still taken at least every "
            "--phase-max-s)."
        ),
    )
    parser.add_argument(
        "--phase-util-threshold-pct",
        type=float,
        default=10.0,
        help="Absolute CPU-utilization change (percent) that ends a phase",
    )
    parser.add_argument(
        "--phase-min-s",
        type=float,
        default=3.0,
        help="Minimum phase length before a utilization change can end it",
    )
    parser.add_argument(
        "--phase-max-s",
        type=float,
        default=30.0,
        help="Hard cap: re-decide at least this often even if utilization is flat",
    )
    parser.add_argument(
        "--phase-baseline-s",
        type=float,
        default=1.0,
        help="Seconds of utilization sampled at phase start to form the baseline",
    )
    parser.add_argument(
        "--phase-recent-s",
        type=float,
        default=0.5,
        help="Seconds of utilization sampled right before the check to compare",
    )
    parser.add_argument("--min-jobs-per-window", type=int, default=5)
    parser.add_argument("--reference-windows", type=int, default=4)
    parser.add_argument("--training-windows", type=int, default=150)
    parser.add_argument("--evaluation-windows", type=int, default=10)
    parser.add_argument("--interleave-reference-every", type=int, default=10)
    parser.add_argument("--reference-window", type=int, default=3)
    parser.add_argument("--paired-evaluation-reference", action="store_true")
    parser.add_argument("--idle-calibration-s", type=float, default=20.0)
    parser.add_argument("--background-power-w", type=float)
    parser.add_argument("--target-idle-power-w", type=float, default=0.58)
    parser.add_argument("--monitor-interval-s", type=float, default=0.20)
    parser.add_argument("--temperature-limit-c", type=float, default=85.0)
    parser.add_argument("--min-frequency-khz", type=int, default=800_000)
    parser.add_argument("--max-frequency-khz", type=int, default=2_100_000)
    parser.add_argument("--fixed-frequency-khz", type=int, default=2_100_000)
    parser.add_argument(
        "--n-frequency-bins",
        type=int,
        default=16,
        help="Number of discrete frequency bins scored by the frequency head",
    )
    parser.add_argument(
        "--fixed-config",
        action="store_true",
        help="Freeze ondemand config to 10 000 us / 95 %%, model decides frequency only",
    )
    parser.add_argument(
        "--free-frequency",
        action="store_true",
        help="Model tunes ondemand sampling_rate/up_threshold only; the governor "
        "floats freely across the full frequency range (no frequency cap)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Config-only mode: agent controls only (sampling_rate, up_threshold). "
        "Frequency is NEVER touched by the agent — the ondemand governor floats "
        "freely across the full range set by prepare_config_only(). "
        "Action space: 50 discrete configs (10 sr × 5 thr). "
        "Implies --n-frequency-bins=1.",
    )
    parser.add_argument(
        "--performance-reference-windows",
        type=int,
        default=0,
        help="Number of performance-governor (always max frequency) baseline "
        "windows to run after the initial ondemand reference phase; their "
        "median EDP is reported as a second evaluation baseline",
    )
    parser.add_argument(
        "--performance-reference-per-lambda-windows",
        type=int,
        default=0,
        help="Performance-governor baseline windows to run at EVERY evaluated "
        "lambda, so each lambda gets its own performance baseline instead of "
        "sharing one pooled median (0 disables the per-lambda sweep)",
    )
    parser.add_argument(
        "--edp-time-metric",
        choices=("response", "service"),
        default=os.environ.get("EDP_TIME_METRIC", "response"),
        help="Time factor inside EDP = power * time**2. 'response' uses the "
        "mean response time (includes queue wait, FCFS sojourn). 'service' "
        "uses the mean service time (pure CPU-on-job time, excludes queue "
        "waiting). Default: response (env EDP_TIME_METRIC overrides).",
    )
    parser.add_argument(
        "--power-basis",
        choices=("target_core", "package"),
        default="package",
        help="Power term inside EDP. 'package' uses the raw package RAPL "
        "power, which is the only basis with enough signal-to-noise to rank "
        "configs (within-lambda CV ~0.15). 'target_core' subtracts a fixed "
        "idle constant for the other package cores; that difference is "
        "dominated by their leakage drift (CV ~0.77) and clamps to the power "
        "floor at low load, so it cannot resolve the few-percent differences "
        "between configs. Default: package.",
    )
    parser.add_argument(
        "--power-floor-w",
        type=float,
        default=0.05,
        help="Lower clamp on the power estimate, in watts. Only matters when "
        "the estimate goes non-positive; such windows are additionally "
        "flagged via power_clamped in the telemetry so they can be excluded.",
    )
    parser.add_argument(
        "--contaminated-power-tol",
        type=float,
        default=0.5,
        help="A window is flagged contaminated when its power deviates from "
        "its lambda's median power by more than this relative fraction. "
        "Contaminated windows are reported separately instead of silently "
        "dragging the mean. Default: 0.5.",
    )
    parser.add_argument(
        "--static-reference-edp",
        type=float,
        default=None,
        help="Static ondemand EDP reference (pre-computed from a separate baseline run). "
        "When set, this fixed value is used as the reference for ALL training reward "
        "computations instead of the rolling interleaved reference. The reward is "
        "log(agent_edp / static_reference_edp).",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("ondemand", "performance"),
        default=None,
        help="Run in baseline collection mode: execute only reference windows (ondemand "
        "or performance governor) for --baseline-windows windows, report the median EDP, "
        "and exit. No training or evaluation is performed.",
    )
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=60,
        help="Number of baseline windows to collect when --baseline-mode is set (default: 60).",
    )
    parser.add_argument(
        "--load-qnets",
        nargs="+",
        action="extend",
        type=Path,
        default=[],
        help="SequentialDDQN checkpoints (save() format) to ensemble for the "
        "evaluation phase; may be passed as one flag with many paths or as "
        "repeated flags. Evaluation windows then use a majority vote across "
        "the greedy (config, frequency) choice of each net, tie-broken by the "
        "highest mean joint Q. Training windows, if any, still use the live agent.",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-replay-size", type=int, default=16)
    parser.add_argument("--updates-per-window", type=int, default=4)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-windows", type=int, default=45)
    parser.add_argument("--package-idle-calibration-s", type=float, default=20.0)
    parser.add_argument("--n-package-cores", type=int)
    parser.add_argument("--rapl-domain", choices=("auto", "core", "package"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-mode",
        choices=("EDP", "FLOPS_WATT", "WEIGHTED"),
        default="EDP",
        help="Reward computation mode. EDP = -log(window_edp / reference_edp) "
        "(or -log(window_edp / static_reference_edp) when --static-reference-edp "
        "is set). FLOPS_WATT = gflops_per_w from the MG1 aggregate. "
        "WEIGHTED = -(energy_weight*log(energy/ref_energy) "
        "+ latency_weight*log(svc/ref_svc)), higher is better.",
    )
    parser.add_argument(
        "--energy-weight", type=float, default=0.5,
        help="Weight on the energy (power*svc) log-ratio term in WEIGHTED reward mode.",
    )
    parser.add_argument(
        "--latency-weight", type=float, default=0.5,
        help="Weight on the service-time log-ratio term in WEIGHTED reward mode.",
    )
    parser.add_argument("--allow-shared-policy", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _postprocess_args(args)
    return args


def _postprocess_args(args: argparse.Namespace) -> None:
    args.lambda_schedule = [
        float(l.strip()) for l in args.lambda_schedule.split(",") if l.strip()
    ]
    args.eval_lambdas = [
        float(l.strip()) for l in args.eval_lambdas.split(",") if l.strip()
    ]
    if args.arrival_rate is None and not args.lambda_schedule:
        raise ValueError("Either --lambda or --lambda-schedule must be specified")
    if args.arrival_rate is not None and args.lambda_schedule:
        raise ValueError("Use --lambda-schedule for universal mode; --lambda is ignored when schedule is set")
    if args.lambda_schedule and not args.eval_lambdas:
        args.eval_lambdas = list(args.lambda_schedule)


def validate_inputs(args: argparse.Namespace) -> Path:
    root = args.mg1_root.expanduser().resolve()
    build = root / "build"
    for expected in (
        build / "Server" / "Server.class",
        build / "client" / "LoadGenerator.class",
    ):
        if not expected.is_file():
            raise FileNotFoundError(
                f"Missing compiled MG1 class {expected}; run the build step first"
            )
    if args.arrival_rate is not None and args.arrival_rate <= 0.0:
        raise ValueError("lambda must be positive")
    if args.duration_s <= args.warmup_s:
        raise ValueError("duration-s must exceed warmup-s")
    if args.window_s <= 0.0:
        raise ValueError("window-s must be positive")
    if not 800_000 <= args.min_frequency_khz < args.max_frequency_khz:
        raise ValueError("invalid frequency range")
    if not args.min_frequency_khz <= args.fixed_frequency_khz <= args.max_frequency_khz:
        raise ValueError("fixed frequency must lie between min and max")
    decision_trigger = getattr(args, "decision_trigger", "fixed")
    if decision_trigger == "phase":
        threshold_pct = getattr(args, "phase_util_threshold_pct", 10.0)
        baseline_s = getattr(args, "phase_baseline_s", 1.0)
        min_s = getattr(args, "phase_min_s", 3.0)
        max_s = getattr(args, "phase_max_s", 30.0)
        if not 0.0 < threshold_pct <= 100.0:
            raise ValueError("phase-util-threshold-pct must be in (0, 100]")
        if not 0.0 < baseline_s < min_s:
            raise ValueError("phase-baseline-s must be below phase-min-s")
        if not min_s < max_s:
            raise ValueError("phase-min-s must be below phase-max-s")
    lambda_schedule = getattr(args, "lambda_schedule", None)
    if lambda_schedule:
        if args.windows_per_lambda <= 0:
            raise ValueError("windows-per-lambda must be positive")
        for lam in lambda_schedule:
            if lam <= 0:
                raise ValueError(f"lambda-schedule values must be positive, got {lam}")
        training_total = len(lambda_schedule) * args.windows_per_lambda
        if args.training_windows != training_total:
            print(f"INFO: overriding training_windows {args.training_windows} -> {training_total} "
                  f"({len(args.lambda_schedule)} lambdas × {args.windows_per_lambda})")
            args.training_windows = training_total
    return root


def make_environment(env: Dict[str, str], telemetry_path: Path) -> Dict[str, str]:
    env = dict(env)
    env["MG1_TELEMETRY_FILE"] = str(telemetry_path)
    return env


def affinity_reserved_mask(cpu: int) -> str:
    return f"{1 << cpu:x}"


def launch_session(
    *,
    root: Path,
    server_cpu: int,
    client_cpu: int,
    arrival_rate: float,
    duration_s: float,
    client_duration_ms: int,
    client_repeat: int,
    telemetry_path: Path,
    log_dir: Path,
) -> Tuple[Mg1Proc, subprocess.Popen]:
    build = root / "build"
    classpath = f"{build}:{root / 'Server' / 'Server' / 'lib'}/*"
    environment = dict(os.environ)
    environment["CLASSPATH"] = classpath
    server_environment = make_environment(environment, telemetry_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = (log_dir / "server_stdout.log").open("wb")
    client_log = (log_dir / "client_stdout.log").open("wb")
    server = subprocess.Popen(
        [
            "java",
            f"-Daffinity.reserved={affinity_reserved_mask(server_cpu)}",
            "Server.Server",
            f"{arrival_rate:.6f}",
            f"{duration_s:.1f}",
        ],
        cwd=str(build),
        env=server_environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    client = subprocess.Popen(
        [
            "taskset",
            "-c",
            str(client_cpu),
            "java",
            "client.LoadGenerator",
            f"{arrival_rate:.6f}",
            f"{client_duration_ms}",
            f"{client_repeat}",
        ],
        cwd=str(build),
        env=environment,
        stdout=client_log,
        stderr=subprocess.STDOUT,
    )
    proc = Mg1Proc()
    proc.server = server
    proc.client = client
    return proc, server


def main() -> int:
    args = parse_args()
    # Support both "--load-qnets a b c" and repeated "--load-qnets a
    # --load-qnets b" spellings: flatten any space-expanded occurrences
    # (entries appended by action="append" may themselves be lists).
    flat: List[Path] = []
    for entry in args.load_qnets:
        if isinstance(entry, (list, tuple)):
            flat.extend(Path(p) for p in entry)
        else:
            flat.append(Path(entry))
    args.load_qnets = flat
    root = validate_inputs(args)
    args.results = args.results.expanduser().resolve()
    # Baseline mode writes a single JSON and exits — no session subdirectory.
    if args.session_name and args.baseline_mode is None:
        args.results = args.results / args.session_name
    args.results.mkdir(parents=True, exist_ok=True)

    config = DDQNConfig(
        state_dim=7,
        n_frequency_bins=args.n_frequency_bins,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        seed=args.seed,
        fmin_khz=args.min_frequency_khz,
        fmax_khz=args.max_frequency_khz,
    )
    if args.config_only:
        ddqn = ConfigOnlyDDQN(config)
        agent = ConfigOnlyWrapper(ddqn)
    else:
        agent = SequentialDDQN(config)
    if args.warmstart:
        agent.warm_start(args.warmstart.expanduser().resolve())

    ensemble_nets: List[SequentialDDQN] = []
    if args.load_qnets and not args.config_only:
        for qnet_path in args.load_qnets:
            qnet_path = qnet_path.expanduser().resolve()
            if not qnet_path.exists():
                raise FileNotFoundError(f"--load-qnets checkpoint not found: {qnet_path}")
            peer = SequentialDDQN(config)
            peer.load(qnet_path)
            ensemble_nets.append(peer)

    cpu = CpuFrequencyController(
        args.target_cpu,
        args.controller_cpu,
        requested_min_khz=args.min_frequency_khz,
        requested_max_khz=args.max_frequency_khz,
        allow_shared_policy=args.allow_shared_policy,
    )
    cpu.assert_process_pinned()
    rapl = RaplDomain.discover(preference=args.rapl_domain)

    if args.config_only:
        cpu.prepare_config_only(initial_sampling_rate_us=10_000, initial_threshold_pct=95)
        print(f"Config-only mode: ondemand governor with full range "
              f"{args.min_frequency_khz}-{args.max_frequency_khz} kHz, "
              f"agent controls (sr, thr) only")
    else:
        cpu.prepare(initial_sampling_rate_us=10_000, initial_threshold_pct=95)

    frequency_centers_khz = agent.frequency_centers_khz
    reference_frequency_id = int(
        np.argmin(np.abs(frequency_centers_khz - args.fixed_frequency_khz))
    )
    reference_config_id = int(
        np.where((ACTION_GRID[:, 0] == 10_000) & (ACTION_GRID[:, 1] == 95))[0][0]
    )
    config_mask = None
    if args.fixed_config:
        config_mask = np.zeros(len(ACTION_GRID), dtype=bool)
        config_mask[reference_config_id] = True
    frequency_mask = None
    if args.free_frequency:
        if args.fixed_config:
            raise ValueError("--free-frequency and --fixed-config are mutually exclusive")
        # With no cap, the only meaningful frequency "choice" is the ceiling:
        # pin the frequency head to the top bin so exploration only spans configs.
        frequency_mask = np.zeros(int(args.n_frequency_bins), dtype=bool)
        frequency_mask[-1] = True

    telemetry_path = args.results / "telemetry.csv"
    eval_lambda_count = (
        len(args.eval_lambdas) if args.lambda_schedule and args.eval_lambdas else 1
    )
    total_windows = _planned_total_windows(args, eval_lambda_count)
    planned_session_s = args.warmup_s + total_windows * args.window_s
    if args.decision_trigger == "phase":
        # Phase windows extend up to phase_max_s; size the client so it never
        # RESETs mid-phase while the RL holds an action.
        planned_session_s = args.warmup_s + total_windows * args.phase_max_s
    if planned_session_s > args.duration_s - 30.0:
        print(
            f"WARNING: planned session ({planned_session_s:.1f}s warmup+windows) "
            f"exceeds --duration-s ({args.duration_s}s) minus the 30s RESET2 "
            "safety margin; the client duration will be extended automatically"
        )
    # The client issues RESET2 (which stops the server) at its duration
    # boundary.  Size it to cover the planned windows plus the min-jobs
    # extensions (~1.3x at light load) and a 60s tail; RESET2 never lands
    # mid-window this way.
    client_budget_s = max(
        args.duration_s,
        planned_session_s * 1.5 + (total_windows * 3.0) + 60.0,
    )
    planned_client_ms = int(round(client_budget_s * 1000))

    preflight = {
        "control_mode": "live_mg1",
        "transition_unit": "mg1_decision_window",
        "decision_trigger": args.decision_trigger,
        "phase_util_threshold_pct": args.phase_util_threshold_pct,
        "phase_min_s": args.phase_min_s,
        "phase_max_s": args.phase_max_s,
        "phase_baseline_s": args.phase_baseline_s,
        "phase_recent_s": args.phase_recent_s,
        "reward": (
            "clip(-energy_weight*log(energy/ref) - latency_weight*log(svc/ref), -5, 5)"
            if args.reward_mode == "WEIGHTED"
            else "clip(-log(window_edp/reference_edp), -5, 5)"
        ),
        "edp_time_metric": args.edp_time_metric,
        "reward_mode_label": args.reward_mode,
        "reward_definition": (
            "target_power_w * mean_service_time_s**2"
            if args.edp_time_metric == "service"
            else "target_power_w * mean_response_time_s**2"
        ),
        "arrival_rate_lambda": args.arrival_rate if args.arrival_rate else args.lambda_schedule[0] if args.lambda_schedule else None,
        "session_duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "window_s": args.window_s,
        "planned_client_duration_ms": planned_client_ms,
        "interleave_reference_every": args.interleave_reference_every,
        "reference_window": args.reference_window,
        "paired_evaluation_reference": args.paired_evaluation_reference,
        "performance_reference_windows": args.performance_reference_windows,
        "performance_reference_per_lambda_windows": (
            args.performance_reference_per_lambda_windows
        ),
        "lambda_schedule": args.lambda_schedule if args.lambda_schedule else None,
        "windows_per_lambda": args.windows_per_lambda if args.lambda_schedule else None,
        "eval_lambdas": args.eval_lambdas if args.lambda_schedule else None,
        "fixed_config": args.fixed_config,
        "free_frequency": args.free_frequency,
        "topology": cpu.topology_summary(),
        "rapl_domain": rapl.name,
        "rapl_path": str(rapl.path),
        "address_10_0_0_2_required": True,
        "affinity_worker_cpu": args.target_cpu,
        "affinity_isolated_cpus_check_note": (
            "server JVM is launched free (no taskset) with "
            "-Daffinity.reserved=<hexmask> so the Affinity library pins its single "
            f"worker to the target CPU {args.target_cpu}"
        ),
        "action_space_size": len(ACTION_GRID) * int(args.n_frequency_bins),
        "effective_action_space_size": (
            1 if args.fixed_config else len(ACTION_GRID) if args.free_frequency else None
        ),
        "agent_actions": [
            "ondemand_sampling_rate_us",
            "ondemand_up_threshold_pct",
            *(() if args.free_frequency else ("ml_frequency_bin",)),
        ],
        "frequency_bins": int(args.n_frequency_bins),
        "reward_mode": args.reward_mode,
        "energy_weight": (args.energy_weight if args.reward_mode == "WEIGHTED" else None),
        "latency_weight": (args.latency_weight if args.reward_mode == "WEIGHTED" else None),
        "frequency_centers_khz": frequency_centers_khz.tolist(),
        "reference_frequency_bin_id": int(reference_frequency_id),
        "reference_frequency_khz": float(frequency_centers_khz[reference_frequency_id]),
        "n_package_cores": args.n_package_cores,
        "agent_config": asdict(config),
        "warmstart": str(args.warmstart.resolve()) if args.warmstart else None,
        "build_root": str(root),
    }
    print(json.dumps(preflight, indent=2))
    if args.preflight_only:
        return 0

    interrupted = False

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    run_log = CsvAppender(args.results / "runs.csv", RUN_FIELDS)
    learning_log = CsvAppender(
        args.results / "learning.csv",
        LEARNING_FIELDS,
    )
    all_rows: List[Dict[str, object]] = []
    proc = Mg1Proc()

    try:
        cpu.prepare(initial_sampling_rate_us=10_000, initial_threshold_pct=95)
        package_idle_power_w, target_idle_power_w, other_cores_idle_power_w = (
            calibrate_package_idle_power(
                cpu,
                rapl,
                args.package_idle_calibration_s,
                args.n_package_cores,
            )
        )
        background_power_w = (
            args.background_power_w
            if args.background_power_w is not None
            else package_idle_power_w
        )
        print(f"Package idle {rapl.name} power: {package_idle_power_w:.4f} W")
        print(f"Target core idle estimate: {target_idle_power_w:.4f} W")
        print(f"Other cores idle estimate: {other_cores_idle_power_w:.4f} W")

        telemetry = TelemetryReader(telemetry_path)
        telemetry.reset()

        initial_lambda = (
            args.lambda_schedule[0] if args.lambda_schedule else args.arrival_rate
        )
        client_duration_ms = planned_client_ms
        server_duration_s = args.duration_s
        proc, _ = launch_session(
            root=root,
            server_cpu=args.target_cpu,
            client_cpu=args.client_cpu,
            arrival_rate=initial_lambda,
            duration_s=server_duration_s,
            client_duration_ms=client_duration_ms,
            client_repeat=args.client_repeat,
            telemetry_path=telemetry_path,
            log_dir=args.results,
        )
        print(f"MG1 server pid={proc.server.pid if proc.server else None} (cpu {args.target_cpu})")
        print(f"MG1 client pid={proc.client.pid if proc.client else None} (cpu {args.client_cpu})")

        # Let the client reset, warm up, and reach queue steady state.
        deadline = time.monotonic() + args.warmup_s
        while time.monotonic() < deadline:
            if not proc.alive():
                raise RuntimeError(
                    "MG1 server or client exited during warmup; "
                    "check server_stdout.log / client_stdout.log"
                )
            time.sleep(0.5)
        telemetry.flush()
        print(f"warmup complete ({args.warmup_s}s), starting reference windows")

        reference_rows: List[Dict[str, object]] = []

        current_lambda: float = (
            args.lambda_schedule[0] if args.lambda_schedule else args.arrival_rate
        )
        lambda_idx = 0
        windows_at_current_lambda = 0

        def switch_lambda(new_lambda: float) -> None:
            nonlocal proc, telemetry, current_lambda, lambda_idx, windows_at_current_lambda
            print(f"  switching lambda: {current_lambda:.1f} -> {new_lambda:.1f}")
            proc.terminate()
            proc = Mg1Proc()
            time.sleep(2.0)
            proc, _ = launch_session(
                root=root,
                server_cpu=args.target_cpu,
                client_cpu=args.client_cpu,
                arrival_rate=new_lambda,
                duration_s=args.duration_s,
                client_duration_ms=planned_client_ms,
                client_repeat=args.client_repeat,
                telemetry_path=telemetry_path,
                log_dir=args.results,
            )
            print(f"MG1 server pid={proc.server.pid if proc.server else None} (cpu {args.target_cpu})")
            print(f"MG1 client pid={proc.client.pid if proc.client else None} (cpu {args.client_cpu})")
            deadline = time.monotonic() + args.warmup_s
            while time.monotonic() < deadline:
                if not proc.alive():
                    raise RuntimeError(
                        "MG1 server or client exited during warmup after lambda switch; "
                        "check server_stdout.log / client_stdout.log"
                    )
                time.sleep(0.5)
            telemetry = TelemetryReader(telemetry_path)
            telemetry.reset()
            telemetry.flush()
            current_lambda = new_lambda
            lambda_idx = next(
                i for i, l in enumerate(args.lambda_schedule)
                if abs(l - new_lambda) < 1e-9
            )
            windows_at_current_lambda = 0
            print(f"  warmup complete at lambda={new_lambda:.1f}")

        def apply_and_measure(
            sampling_rate,
            threshold,
            requested_frequency,
            already_applied=False,
            decision_trigger: str = "",
        ) -> Dict[str, object]:
            if already_applied:
                mapped_frequency = int(requested_frequency)
            else:
                mapped_frequency = cpu.apply_action(sampling_rate, threshold, requested_frequency)
            trigger = decision_trigger or args.decision_trigger
            phase_tracker = None
            if trigger == "phase":
                phase_tracker = PhaseTracker(
                    baseline_s=args.phase_baseline_s,
                    recent_s=args.phase_recent_s,
                    threshold_pct=args.phase_util_threshold_pct,
                    min_s=args.phase_min_s,
                    max_s=args.phase_max_s,
                )
                phase_tracker.reset()
            util_before = cpu_stat(args.target_cpu)
            energy_before = rapl.read()
            start = time.monotonic()
            frequency_samples: List[float] = []
            temperatures: List[float] = []
            window_rows: List[Dict[str, float]] = []
            phase_stat_cursor: Optional[Tuple[int, int]] = util_before
            # Hold the chosen action for at least window_s, extending until the
            # minimum job count arrives (or a hard cap is hit) so empty windows
            # do not poison the MRT/EDP estimate under light load.
            hard_cap_s = max(args.window_s * 4.0, 120.0)
            while True:
                time.sleep(args.monitor_interval_s)
                if not proc.alive():
                    raise RuntimeError(
                        "MG1 server or client exited mid-window; "
                        "check server_stdout.log / client_stdout.log"
                    )
                frequency_samples.append(float(cpu.current_frequency_khz()))
                temperature = read_max_temperature_c()
                if temperature is not None:
                    temperatures.append(float(temperature))
                    if temperature >= args.temperature_limit_c:
                        raise RuntimeError(
                            f"Temperature {temperature:.1f} C above safety limit"
                        )
                elapsed = time.monotonic() - start
                rows, _ = telemetry.read()
                window_rows.extend(rows)
                if trigger == "phase":
                    if phase_tracker is not None:
                        util_now = cpu_stat(args.target_cpu)
                        phase_tracker.add_sample(
                            elapsed, utilization_between(phase_stat_cursor, util_now)
                        )
                        phase_stat_cursor = util_now
                    if phase_tracker is not None and phase_tracker.should_end(elapsed):
                        break
                elif elapsed >= args.window_s:
                    completed = len(window_rows)
                    if completed >= args.min_jobs_per_window or elapsed >= hard_cap_s:
                        break
            energy_after = rapl.read()
            util_after = cpu_stat(args.target_cpu)
            window_time = energy_after.timestamp - energy_before.timestamp
            domain_energy = rapl.joules_between(energy_before, energy_after)
            domain_power = domain_energy / max(window_time, 1e-9)
            estimated_target_power, power_clamped = _estimate_window_power(
                domain_power=domain_power,
                other_cores_idle_power_w=other_cores_idle_power_w,
                basis=args.power_basis,
                floor_w=args.power_floor_w,
            )
            aggregate = aggregate_rows(window_rows)
            valid = (
                proc.alive()
                and aggregate is not None
                and int(aggregate["jobs_completed"]) >= args.min_jobs_per_window
            )
            if not aggregate:
                aggregate = {
                    "jobs_completed": 0,
                    "mean_response_time_s": float("nan"),
                    "mean_service_time_s": float("nan"),
                    "mean_wait_time_s": float("nan"),
                    "total_flops": 0.0,
                    "mean_flops_per_job": 0.0,
                }
            if aggregate.get("jobs_completed", 0) > 0:
                power_w = max(float(estimated_target_power), 1e-9)
                seconds = max(float(window_time), 1e-9)
                total_flops = float(aggregate.get("total_flops", 0.0))
                gflops_per_s = (total_flops / seconds) / 1e9
                aggregate["gflops_per_s"] = gflops_per_s
                aggregate["gflops_per_w"] = gflops_per_s / power_w
            else:
                aggregate["gflops_per_s"] = float("nan")
                aggregate["gflops_per_w"] = float("nan")
            return {
                "aggregate": aggregate,
                "valid": valid,
                "mapped_frequency": mapped_frequency,
                "utilization": utilization_between(util_before, util_after),
                "mean_measured_frequency_khz": (
                    float(np.mean(frequency_samples))
                    if frequency_samples
                    else float(cpu.current_frequency_khz())
                ),
                "domain_energy_j": domain_energy,
                "domain_power_w": domain_power,
                "other_cores_idle_power_w": other_cores_idle_power_w,
                "estimated_target_power_w": estimated_target_power,
                "power_clamped": power_clamped,
                "power_basis": args.power_basis,
        "power_floor_w": args.power_floor_w,
        "contaminated_power_tol": args.contaminated_power_tol,
                "temperature_c": max(temperatures) if temperatures else None,
                "window_seconds": window_time,
                "decision_trigger": trigger,
                "phase_baseline_utilization": (
                    phase_tracker.baseline_utilization if phase_tracker else None
                ),
                "phase_recent_utilization": (
                    phase_tracker.recent_utilization if phase_tracker else None
                ),
            }

        def perform_reference_window(
            run_number: str, phase: str = "reference", terminal: bool = False,
            done: bool = False,
        ) -> Dict[str, object]:
            mapped_cap = cpu.apply_baseline_ondemand(10_000, 95, args.fixed_frequency_khz)
            measured = apply_and_measure(
                10_000, 95, mapped_cap, already_applied=True, decision_trigger="fixed"
            )
            aggregate = measured["aggregate"]
            edp = _window_edp(measured, aggregate, args.edp_time_metric)
            row = _base_row(
                agent=agent,
                run_number=run_number,
                phase=phase,
                config_id=reference_config_id,
                sampling_rate=10_000,
                threshold=95,
                frequency_id=reference_frequency_id,
                requested_frequency=float(frequency_centers_khz[reference_frequency_id]),
                mapped_frequency=mapped_cap,
                measured=measured,
                aggregate=aggregate,
                edp=edp,
                reward=0.0,
                done=done or terminal,
                reference_candidate=True,
                reference_edp=None,
                reference_source="reference_window",
                reference_samples=0,
                epsilon=0.0,
                exploratory=False,
                greedy_config_id="",
                greedy_frequency_bin_id="",
                chosen_q="",
                max_q="",
                training_loss=float("nan"),
                training_updates=0,
                requested_lambda=current_lambda,
            )
            run_log.append(row)
            all_rows.append(row)
            if measured["valid"]:
                reference_rows.append(row)
                print(
                    f"reference {run_number}: jobs={aggregate['jobs_completed']} "
                    f"MRT={aggregate['mean_response_time_s']:.4f}s "
                    f"P={measured['estimated_target_power_w']:.3f}W "
                    f"EDP={edp:.6g}"
                )
            return row

        performance_rows: List[Dict[str, object]] = []

        def perform_performance_window(run_number: str) -> Dict[str, object]:
            """Baseline window with the performance governor (always max freq).

            Does NOT feed the rolling ondemand reward reference; its rows are
            recorded with phase ``performance_reference`` only.
            """
            mapped_cap = cpu.apply_performance_governor(args.fixed_frequency_khz)
            measured = apply_and_measure(
                10_000, 95, mapped_cap, already_applied=True, decision_trigger="fixed"
            )
            aggregate = measured["aggregate"]
            edp = _window_edp(measured, aggregate, args.edp_time_metric)
            row = _base_row(
                agent=agent,
                run_number=run_number,
                phase="performance_reference",
                config_id=reference_config_id,
                sampling_rate=10_000,
                threshold=95,
                frequency_id=reference_frequency_id,
                requested_frequency=float(frequency_centers_khz[reference_frequency_id]),
                mapped_frequency=mapped_cap,
                measured=measured,
                aggregate=aggregate,
                edp=edp,
                reward=0.0,
                done=False,
                reference_candidate=False,
                reference_edp=None,
                reference_source="performance_governor",
                reference_samples=0,
                epsilon=0.0,
                exploratory=False,
                greedy_config_id="",
                greedy_frequency_bin_id="",
                chosen_q="",
                max_q="",
                training_loss=float("nan"),
                training_updates=0,
                requested_lambda=current_lambda,
            )
            run_log.append(row)
            all_rows.append(row)
            if measured["valid"]:
                performance_rows.append(row)
                print(
                    f"performance-ref {run_number}: jobs={aggregate['jobs_completed']} "
                    f"MRT={aggregate['mean_response_time_s']:.4f}s "
                    f"P={measured['estimated_target_power_w']:.3f}W "
                    f"EDP={edp:.6g}"
                )
            return row

        def reference_statistics() -> Tuple[float, float, float, float]:
            window = reference_rows[-args.reference_window :]
            edps = [
                float(row["edp_w_s2"])
                for row in window
                if isinstance(row.get("edp_w_s2"), (int, float))
                and np.isfinite(float(row["edp_w_s2"]))
            ]
            powers = [
                float(row["estimated_target_power_w"])
                for row in window
                if isinstance(row.get("estimated_target_power_w"), (int, float))
            ]
            svcs = [
                float(row["mean_service_time_s"])
                for row in window
                if isinstance(row.get("mean_service_time_s"), (int, float))
                and np.isfinite(float(row["mean_service_time_s"]))
            ]
            if not edps:
                raise RuntimeError("No finite EDP reference samples")
            ref_power = float(np.median(powers)) if powers else float("nan")
            ref_svc = float(np.median(svcs)) if svcs else float("nan")
            return (
                float(np.median(edps)),
                ref_power,
                ref_svc,
                ref_power * ref_svc,  # reference energy (power * service_time)
            )

        # ---- Baseline mode: collect reference EDPs and exit ----
        # Must run here (after perform_reference_window / perform_performance_window
        # are defined) so the helper functions are in scope.  Baseline mode exits
        # the whole program, so it only needs to be checked once, not per training
        # window.
        if args.baseline_mode is not None:
            baseline_rows = []
            is_perf = args.baseline_mode == "performance"
            target_gov = "performance" if is_perf else "ondemand"
            print(f"\n=== Baseline mode: {target_gov}, {args.baseline_windows} windows ===", flush=True)
            for i in range(1, args.baseline_windows + 1):
                if is_perf:
                    row = perform_performance_window(f"baseline_{i}")
                else:
                    row = perform_reference_window(f"baseline_{i}", phase="baseline_ondemand")
                baseline_rows.append(row)
            edps = []
            for r in baseline_rows:
                v = r.get("edp_w_s2")
                if v and np.isfinite(float(v)):
                    edps.append(float(v))
                    print(f"  window: EDP={v:.6g}", flush=True)
            if not edps:
                print("ERROR: No valid baseline EDPs collected", flush=True)
                return 1
            median_edp = float(np.median(edps))
            mean_edp = float(np.mean(edps))
            std_edp = float(np.std(edps))
            print(f"\n=== Baseline result: {target_gov} ===", flush=True)
            print(f"  windows: {len(edps)}", flush=True)
            print(f"  median EDP: {median_edp:.6f} W*s^2", flush=True)
            print(f"  mean EDP:   {mean_edp:.6f} W*s^2", flush=True)
            print(f"  std EDP:    {std_edp:.6f} W*s^2", flush=True)
            import json as _json
            result = {
                "mode": args.baseline_mode,
                "lambda": args.arrival_rate or (args.lambda_schedule[0] if args.lambda_schedule else None),
                "windows": len(edps),
                "median_edp": median_edp,
                "mean_edp": mean_edp,
                "std_edp": std_edp,
                "all_edps": edps,
            }
            with open(args.results / "baseline_result.json", "w") as f:
                _json.dump(result, f, indent=2)
            print(f"  Saved to {args.results / 'baseline_result.json'}", flush=True)
            return 0

        for run_number in range(1, args.reference_windows + 1):
            perform_reference_window(f"initial_{run_number}", terminal=run_number == args.reference_windows)

        if not reference_rows:
            raise RuntimeError("No valid reference queue window completed")
        reference_edp, reference_power, reference_svc, reference_energy = reference_statistics()
        initial_reference_edp = reference_edp
        initial_reference_power = reference_power
        reference_svc_ref = reference_svc
        reference_energy_ref = reference_energy
        print(f"Reference median EDP: {reference_edp:.6g} W*s^2")

        for perf_number in range(1, args.performance_reference_windows + 1):
            perform_performance_window(f"performance_{perf_number}")
        if performance_rows:
            perf_median = float(
                np.median([float(r["edp_w_s2"]) for r in performance_rows])
            )
            print(f"Performance-governor median EDP: {perf_median:.6g} W*s^2")

        last_reference = reference_rows[-1]
        state = _state_from_measurement(
            agent,
            last_reference,
            int(last_reference["sampling_rate_us"]),
            int(last_reference["threshold_pct"]),
            reference_power,
            progress=0.0,
        )

        training_totals = max(args.training_windows, 1)
        train_rows: List[Dict[str, object]] = []
        for run_number in range(1, args.training_windows + 1):
            # ---- Baseline mode: collect reference EDPs and exit ----
            # Checked on every training window so a --baseline-mode run can exit
            # early without fighting the interleave-reference gating.
            if args.baseline_mode is not None:
                baseline_rows = []
                is_perf = args.baseline_mode == "performance"
                target_gov = "performance" if is_perf else "ondemand"
                print(f"\n=== Baseline mode: {target_gov}, {args.baseline_windows} windows ===", flush=True)
                for i in range(1, args.baseline_windows + 1):
                    if is_perf:
                        row = perform_performance_window(f"baseline_{i}")
                    else:
                        row = perform_reference_window(f"baseline_{i}", phase="baseline_ondemand")
                    baseline_rows.append(row)
                edps = []
                for r in baseline_rows:
                    v = r.get("edp_w_s2")
                    if v and np.isfinite(float(v)):
                        edps.append(float(v))
                        print(f"  window: EDP={v:.6g}", flush=True)
                if not edps:
                    print("ERROR: No valid baseline EDPs collected", flush=True)
                    return 1
                median_edp = float(np.median(edps))
                mean_edp = float(np.mean(edps))
                std_edp = float(np.std(edps))
                print(f"\n=== Baseline result: {target_gov} ===", flush=True)
                print(f"  windows: {len(edps)}", flush=True)
                print(f"  median EDP: {median_edp:.6f} W*s^2", flush=True)
                print(f"  mean EDP:   {mean_edp:.6f} W*s^2", flush=True)
                print(f"  std EDP:    {std_edp:.6f} W*s^2", flush=True)
                import json as _json
                result = {
                    "mode": args.baseline_mode,
                    "lambda": args.arrival_rate or (args.lambda_schedule[0] if args.lambda_schedule else None),
                    "windows": len(edps),
                    "median_edp": median_edp,
                    "mean_edp": mean_edp,
                    "std_edp": std_edp,
                    "all_edps": edps,
                }
                with open(args.results / "baseline_result.json", "w") as f:
                    _json.dump(result, f, indent=2)
                print(f"  Saved to {args.results / 'baseline_result.json'}", flush=True)
                return 0

            if args.lambda_schedule:
                if windows_at_current_lambda >= args.windows_per_lambda:
                    next_idx = (lambda_idx + 1) % len(args.lambda_schedule)
                    switch_lambda(args.lambda_schedule[next_idx])
            if (
                args.interleave_reference_every > 0
                and run_number > 1
                and (run_number - 1) % args.interleave_reference_every == 0
            ):
                interleaved = perform_reference_window(
                    run_number, phase="reference_interleaved", done=False
                )
                reference_edp, reference_power, reference_svc, reference_energy = reference_statistics()
                reference_svc_ref = reference_svc
                reference_energy_ref = reference_energy
                state = _state_from_measurement(
                    agent,
                    interleaved,
                    10_000,
                    95,
                    reference_power,
                    progress=run_number / training_totals,
                    )

            # ---- Baseline mode: collect reference EDPs and exit ----
            if args.baseline_mode is not None:
                        baseline_rows = []
                        is_perf = args.baseline_mode == "performance"
                        target_gov = "performance" if is_perf else "ondemand"
                        print(f"\n=== Baseline mode: {target_gov}, {args.baseline_windows} windows ===", flush=True)
                        for i in range(1, args.baseline_windows + 1):
                            if is_perf:
                                row = perform_performance_window(f"baseline_{i}")
                            else:
                                row = perform_reference_window(f"baseline_{i}", phase="baseline_ondemand")
                            baseline_rows.append(row)
                        edps = []
                        for r in baseline_rows:
                            v = r.get("edp_w_s2")
                            if v and np.isfinite(float(v)):
                                edps.append(float(v))
                                print(f"  window: EDP={v:.6g}", flush=True)
                        if not edps:
                            print("ERROR: No valid baseline EDPs collected", flush=True)
                            return 1
                        median_edp = float(np.median(edps))
                        mean_edp = float(np.mean(edps))
                        std_edp = float(np.std(edps))
                        print(f"\n=== Baseline result: {target_gov} ===", flush=True)
                        print(f"  windows: {len(edps)}", flush=True)
                        print(f"  median EDP: {median_edp:.6f} W*s^2", flush=True)
                        print(f"  mean EDP:   {mean_edp:.6f} W*s^2", flush=True)
                        print(f"  std EDP:    {std_edp:.6f} W*s^2", flush=True)
                        import json as _json
                        result = {
                            "mode": args.baseline_mode,
                            "lambda": args.arrival_rate or (args.lambda_schedule[0] if args.lambda_schedule else None),
                            "windows": len(edps),
                            "median_edp": median_edp,
                            "mean_edp": mean_edp,
                            "std_edp": std_edp,
                            "all_edps": edps,
                        }
                        with open(args.results / "baseline_result.json", "w") as f:
                            _json.dump(result, f, indent=2)
                        print(f"  Saved to {args.results / 'baseline_result.json'}", flush=True)
                        return 0

            epsilon = epsilon_by_step(
                agent.environment_steps,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_windows,
            )
            # Log decision (frequency_id=0 in config-only mode since we don't
            # control frequency — the ondemand governor floats freely)
            config_id, frequency_id, exploratory, greedy_config_id, greedy_frequency_id, chosen_q, max_q = (
                agent.action(state, epsilon, config_mask=config_mask, frequency_mask=frequency_mask)
            )
            sampling_rate = int(ACTION_GRID[config_id, 0])
            threshold = int(ACTION_GRID[config_id, 1])
            if args.config_only:
                cpu.set_config_only(sampling_rate, threshold)
                measured = apply_and_measure(
                    sampling_rate, threshold, 0, already_applied=True
                )
                requested_frequency = 0.0
                frequency_id = 0
            elif args.free_frequency:
                requested_frequency = float(
                    cpu.apply_baseline_ondemand(
                        sampling_rate, threshold, args.fixed_frequency_khz
                    )
                )
                measured = apply_and_measure(
                    sampling_rate, threshold, requested_frequency, already_applied=True
                )
            else:
                requested_frequency = float(frequency_centers_khz[frequency_id])
                measured = apply_and_measure(sampling_rate, threshold, requested_frequency)
            aggregate = measured["aggregate"]
            edp = _window_edp(measured, aggregate, args.edp_time_metric)
            if args.reward_mode == "FLOPS_WATT":
                reward = float(aggregate.get("gflops_per_w", 0.0))
            elif args.reward_mode == "WEIGHTED":
                p_w = max(float(measured["estimated_target_power_w"]), 1e-9)
                svc = float(aggregate.get("mean_service_time_s", float("nan")))
                energy = p_w * svc
                rew_e = -np.log(energy / reference_energy) if reference_energy > 0 else -1e6
                rew_s = -np.log(svc / reference_svc) if (reference_svc > 0 and svc > 0) else -1e6
                reward = float(np.clip(args.energy_weight * rew_e
                                       + args.latency_weight * rew_s, -5.0, 5.0))
            else:
                reward = -np.log(edp / reference_edp) if reference_edp > 0 else -float(edp)
            if not np.isfinite(reward):
                # Soft sentinel (-5, the same bound as the reward clip) instead of
                # -1e6: a single invalid window must not explode the Q-targets /
                # network (was causing catastrophic MSE divergence).
                reward = -5.0
            next_state = _state_from_measurement(
                agent,
                {**_measurement_as_candidate_row(measured, aggregate, edp),
                 "sampling_rate_us": sampling_rate,
                 "threshold_pct": threshold,
                 "estimated_target_power_w": measured["estimated_target_power_w"]},
                sampling_rate,
                threshold,
                reference_power,
                progress=run_number / training_totals,
            )
            episode_id, episode_step, done = training_episode_position(
                run_number, args.training_windows, args.interleave_reference_every
            )
            agent.observe(state, config_id, frequency_id, reward, next_state, done)
            update_result = agent.update(args.updates_per_window)
            row = _base_row(
                agent=agent,
                run_number=run_number,
                phase="train",
                config_id=config_id,
                sampling_rate=sampling_rate,
                threshold=threshold,
                frequency_id=frequency_id,
                requested_frequency=requested_frequency,
                mapped_frequency=measured["mapped_frequency"],
                measured=measured,
                aggregate=aggregate,
                edp=edp,
                reward=reward,
                done=done,
                reference_candidate=False,
                reference_edp=reference_edp,
                reference_source=(
                    "rolling_interleaved"
                    if args.interleave_reference_every > 0
                    else "initial_median"
                ),
                reference_samples=min(len(reference_rows), args.reference_window),
                epsilon=epsilon,
                exploratory=exploratory,
                greedy_config_id=greedy_config_id,
                greedy_frequency_bin_id=greedy_frequency_id,
                chosen_q=chosen_q,
                max_q=max_q,
                training_loss=update_result["loss"],
                training_updates=update_result["updates"],
                training_metrics=update_result,
                episode_id=episode_id,
                episode_step=episode_step,
                requested_lambda=current_lambda,
            )
            attach_state(row, "state", state)
            attach_state(row, "next_state", next_state)
            run_log.append(row)
            all_rows.append(row)
            train_rows.append(row)
            for metric_row in update_result.get("history", []):
                learning_log.append(
                    {
                        **metric_row,
                        "run": run_number,
                        "episode_id": episode_id,
                        "episode_step": episode_step,
                        "epsilon": epsilon,
                        "exploratory": exploratory,
                        "reward": reward,
                        "utilization": measured["utilization"],
                        "edp_w_s2": edp if np.isfinite(edp) else "",
                    }
                )
            state = next_state
            agent.save(args.results / "mg1_ddqn_checkpoint.npz")
            windows_at_current_lambda += 1
            lam_tag = f"lam={current_lambda:.1f} " if args.lambda_schedule else ""
            print(
                f"train {run_number}/{training_totals}: {lam_tag}cfg={config_id} "
                f"sr={sampling_rate} thr={threshold} "
                f"bin={frequency_id} req={requested_frequency:.0f}kHz "
                f"jobs={aggregate['jobs_completed']} MRT={aggregate['mean_response_time_s']:.4f}s "
                f"EDP={edp:.6g} reward={reward:+.4f} replay={agent.replay.size} "
                f"episode={episode_id}:{episode_step}"
            )
            if args.decision_trigger == "phase":
                print(
                    f"    phase: {measured['window_seconds']:.1f}s "
                    f"baseline_util={measured.get('phase_baseline_utilization'):.3f} "
                    f"recent_util={measured.get('phase_recent_utilization'):.3f}"
                )

        evaluation_rows: List[Dict[str, object]] = []
        evaluation_reference_rows: List[Dict[str, object]] = []
        eval_lambdas = args.eval_lambdas if args.lambda_schedule else [current_lambda]
        for eval_lam in eval_lambdas:
            if args.lambda_schedule and abs(eval_lam - current_lambda) > 1e-9:
                switch_lambda(eval_lam)
            eval_windows_tag = f"eval@{eval_lam:.1f}"
            # Per-lambda performance-governor baseline. Without this the only
            # performance windows are the ones taken at the initial lambda, so a
            # single pooled median was being compared against every lambda.
            for perf_number in range(1, args.performance_reference_per_lambda_windows + 1):
                perform_performance_window(
                    f"performance_{eval_lam:.1f}_{perf_number}"
                )
            for run_number in range(1, args.evaluation_windows + 1):
                    if args.paired_evaluation_reference:
                        paired = perform_reference_window(
                            f"evaluation_{run_number}",
                            phase="reference_interleaved",
                            done=False,
                        )
                        evaluation_reference_rows.append(paired)
                        paired_edp = paired.get("edp_w_s2")
                        if not isinstance(paired_edp, (int, float)) or not np.isfinite(float(paired_edp)):
                            raise RuntimeError(
                                f"Paired evaluation reference EDP invalid: {paired_edp!r}"
                            )
                        reference_edp = float(paired_edp)
                        reference_power = float(paired["estimated_target_power_w"])
                        state = _state_from_measurement(
                            agent,
                            paired,
                            10_000,
                            95,
                            reference_power,
                            progress=1.0,
                        )

                    if ensemble_nets:
                        (
                            config_id,
                            frequency_id,
                            greedy_config_id,
                            greedy_frequency_id,
                            chosen_q,
                            max_q,
                            ensemble_vote_count,
                        ) = _ensemble_decision(ensemble_nets, state, config_mask, frequency_mask)
                        exploratory = False
                    else:
                        (
                            config_id,
                            frequency_id,
                            exploratory,
                            greedy_config_id,
                            greedy_frequency_id,
                            chosen_q,
                            max_q,
                        ) = agent.action(state, 0.0, config_mask=config_mask, frequency_mask=frequency_mask)
                    sampling_rate = int(ACTION_GRID[config_id, 0])
                    threshold = int(ACTION_GRID[config_id, 1])
                    if args.config_only:
                        cpu.set_config_only(sampling_rate, threshold)
                        measured = apply_and_measure(
                            sampling_rate, threshold, 0, already_applied=True
                        )
                        requested_frequency = 0.0
                        frequency_id = 0
                    elif args.free_frequency:
                        requested_frequency = float(
                            cpu.apply_baseline_ondemand(
                                sampling_rate, threshold, args.fixed_frequency_khz
                            )
                        )
                        measured = apply_and_measure(
                            sampling_rate, threshold, requested_frequency, already_applied=True
                        )
                    else:
                        requested_frequency = float(frequency_centers_khz[frequency_id])
                        measured = apply_and_measure(sampling_rate, threshold, requested_frequency)
                    aggregate = measured["aggregate"]
                    edp = _window_edp(measured, aggregate, args.edp_time_metric)
                    done = run_number == args.evaluation_windows
                    row = _base_row(
                        agent=agent,
                        run_number=run_number,
                        phase="evaluation",
                        config_id=config_id,
                        sampling_rate=sampling_rate,
                        threshold=threshold,
                        frequency_id=frequency_id,
                        requested_frequency=requested_frequency,
                        mapped_frequency=measured["mapped_frequency"],
                        measured=measured,
                        aggregate=aggregate,
                        edp=edp,
                        reward=edp_reward(edp, reference_edp),
                        done=done,
                        reference_candidate=False,
                        reference_edp=reference_edp,
                        reference_source=(
                            "paired_interleaved"
                            if args.paired_evaluation_reference
                            else (
                                "rolling_interleaved"
                                if args.interleave_reference_every > 0
                                else "initial_median"
                            )
                        ),
                        reference_samples=min(
                            1 if args.paired_evaluation_reference else len(reference_rows),
                            args.reference_window,
                        ),
                        epsilon=0.0,
                        exploratory=exploratory,
                        greedy_config_id=greedy_config_id,
                        greedy_frequency_bin_id=greedy_frequency_id,
                        chosen_q=chosen_q,
                        max_q=max_q,
                        training_loss=float("nan"),
                        training_updates=0,
                        requested_lambda=current_lambda,
                    )
                    attach_state(row, "state", state)
                    if ensemble_nets:
                        row["ensemble_vote_count"] = ensemble_vote_count
                    run_log.append(row)
                    all_rows.append(row)
                    evaluation_rows.append(row)
                    state = _state_from_measurement(
                        agent,
                        row,
                        sampling_rate,
                        threshold,
                        reference_power,
                        progress=1.0,
                    )
                    print(
                        f"{eval_windows_tag} {run_number}: cfg={config_id} sr={sampling_rate} "
                        f"thr={threshold} bin={frequency_id} req={requested_frequency:.0f}kHz "
                        f"jobs={aggregate['jobs_completed']} MRT={aggregate['mean_response_time_s']:.4f}s "
                        f"EDP={edp:.6g} delta={100 * (edp / reference_edp - 1.0):+.2f}%"
                    )

        proc.terminate()
        telemetry_reader = telemetry.read()
        summary = {
            **preflight,
            "other_cores_idle_power_w": other_cores_idle_power_w,
            "background_power_w": background_power_w,
            "reference_windows_valid": len(reference_rows),
            "interleaved_reference_windows": sum(
                row["phase"] == "reference_interleaved" for row in all_rows
            ),
            "reference_edp_w_s2": initial_reference_edp,
            "reference_target_power_w": initial_reference_power,
            "final_reference_edp_w_s2": reference_edp,
            "training_windows_valid": sum(row["phase"] == "train" for row in all_rows),
            "evaluation_windows_valid": len(evaluation_rows),
            "evaluation_mean_edp_w_s2": (
                float(np.mean([row["edp_w_s2"] for row in evaluation_rows]))
                if evaluation_rows
                else None
            ),
            "evaluation_mean_delta_vs_reference": (
                float(
                    np.mean(
                        [
                            row["edp_w_s2"] / float(row["reference_edp_w_s2"]) - 1.0
                            for row in evaluation_rows
                        ]
                    )
                )
                if evaluation_rows
                else None
            ),
            "evaluation_reference_median_edp_w_s2": (
                float(
                    np.median(
                        [
                            float(row["reference_edp_w_s2"])
                            for row in evaluation_rows
                        ]
                    )
                )
                if evaluation_rows
                else None
            ),
            "evaluation_mean_gain_vs_reference_median": (
                _gain_vs_baseline(evaluation_rows) if evaluation_rows else None
            ),
            "performance_reference_windows_valid": len(performance_rows),
            "performance_reference_median_edp_w_s2": (
                float(np.median([float(r["edp_w_s2"]) for r in performance_rows]))
                if performance_rows
                else None
            ),
            "evaluation_mean_gain_vs_performance": (
                _gain_vs_baseline(evaluation_rows, performance_rows)
                if evaluation_rows and performance_rows
                else None
            ),
            "evaluation_mean_gflops_per_w": (
                float(
                    np.mean(
                        [
                            float(r["gflops_per_w"])
                            for r in evaluation_rows
                            if isinstance(r.get("gflops_per_w"), (int, float))
                            and np.isfinite(float(r["gflops_per_w"]))
                        ]
                    )
                )
                if evaluation_rows
                else None
            ),
            "evaluation_reference_median_gflops_per_w": (
                float(
                    np.median(
                        [
                            float(r["gflops_per_w"])
                            for r in evaluation_reference_rows
                            if isinstance(r.get("gflops_per_w"), (int, float))
                            and np.isfinite(float(r["gflops_per_w"]))
                        ]
                    )
                )
                if evaluation_reference_rows
                else None
            ),
            "evaluation_mean_gflops_gain_vs_reference_median": (
                _gain_vs_baseline_gflops(evaluation_rows, evaluation_reference_rows)
                if evaluation_rows and evaluation_reference_rows
                else None
            ),
            "performance_reference_mean_gflops_per_w": (
                float(
                    np.median(
                        [
                            float(r["gflops_per_w"])
                            for r in performance_rows
                            if isinstance(r.get("gflops_per_w"), (int, float))
                            and np.isfinite(float(r["gflops_per_w"]))
                        ]
                    )
                )
                if performance_rows
                else None
            ),
            "evaluation_mean_gflops_gain_vs_performance": (
                _gain_vs_baseline_gflops(evaluation_rows, performance_rows)
                if evaluation_rows and performance_rows
                else None
            ),
            "evaluation_per_lambda_edp": (
                {
                    float(lam): float(
                        np.mean([
                            float(r["edp_w_s2"]) for r in evaluation_rows
                            if abs(float(r["requested_lambda"]) - lam) < 1e-9
                        ])
                    )
                    for lam in eval_lambdas
                }
                if args.lambda_schedule and evaluation_rows else None
            ),
            # ---- per-lambda baselines -------------------------------------
            # The pooled keys above mix every lambda into one median, so they
            # cannot be compared against a single lambda's EDP. These are the
            # per-lambda values that must be used for per-lambda reporting.
            "baselines_pooled_across_lambdas": True,
            **(
                _per_lambda_baseline_summary(
                    evaluation_rows,
                    performance_rows,
                    eval_lambdas,
                    args.contaminated_power_tol,
                )
                if args.lambda_schedule and evaluation_rows
                else _per_lambda_baseline_summary([], [], [], args.contaminated_power_tol)
            ),
            "environment_steps": agent.environment_steps,
            "training_steps": agent.training_steps,
            "load_qnets": [str(q) for q in args.load_qnets],
            "ensemble_unanimous": (
                sum(1 for r in evaluation_rows if r.get("ensemble_vote_count", 0) == len(args.load_qnets))
                if args.load_qnets and evaluation_rows
                else None
            ),
            "telemetry_rows_seen": telemetry_reader[-1] if telemetry_reader else 0,
            "arrival_rate": args.arrival_rate if args.arrival_rate else args.lambda_schedule,
            "interrupted": interrupted,
        }
        (args.results / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        agent.save(args.results / "mg1_ddqn_final.npz")
        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            cpu.restore()
        except Exception as restore_error:
            print(f"WARNING: CPUFreq restore failed: {restore_error}", file=sys.stderr)


def _window_edp(
    measured: Dict[str, object],
    aggregate: Dict[str, float],
    time_metric: str = "response",
) -> float:
    key = "mean_service_time_s" if time_metric == "service" else "mean_response_time_s"
    t = float(aggregate[key])
    if not np.isfinite(t) or t <= 0.0:
        return float("nan")
    return float(measured["estimated_target_power_w"]) * t * t


def _ensemble_decision(nets, state, config_mask, frequency_mask):
    return ensemble_decision(nets, state, config_mask, frequency_mask)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _contaminated_mask(rows, tol: float, power_key: str = "domain_power_w"):
    """Flag windows whose power is an outlier for their own lambda.

    RAPL occasionally reports a value far outside the lambda's distribution --
    observed up to 19x the median on an otherwise identical workload.  Those
    windows are not rare enough to ignore and common enough to destroy a mean,
    so they are marked rather than silently included or silently dropped.
    """
    mask = [False] * len(rows)
    powers = [r.get(power_key) for r in rows]
    usable_idx = [i for i, p in enumerate(powers) if _finite(p) and float(p) > 0]
    if len(usable_idx) < 3 or not (tol > 0):
        return mask
    median_power = float(np.median([float(powers[i]) for i in usable_idx]))
    if median_power <= 0:
        return mask
    for i in usable_idx:
        mask[i] = abs(float(powers[i]) - median_power) / median_power > tol
    return mask


def _robust_gain(gains, trim: float = 0.1) -> Dict[str, object]:
    """Aggregate per-window gains without letting a few spikes decide.

    Reports the median as the headline (unbeatable by outliers) and a
    symmetric trimmed mean alongside it, so a result that only looks good
    because of a heavy tail is visible as such.
    """
    values = np.asarray([g for g in gains if np.isfinite(g)], dtype=float)
    if values.size == 0:
        return None
    ordered = np.sort(values)
    if ordered.size >= 10 and 0.0 <= trim < 0.5:
        cut = int(np.floor(ordered.size * trim))
        kept = ordered[cut: ordered.size - cut] if cut else ordered
    else:
        kept = ordered
    return {
        "median_gain": float(np.median(values)),
        "trimmed_mean_gain": float(np.mean(kept)),
        "mean_gain": float(np.mean(values)),
        "win_rate": float(np.mean(values > 0.0)),
        "iqr_gain": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "min_gain": float(ordered[0]),
        "max_gain": float(ordered[-1]),
        "n_windows": float(values.size),
    }


def _per_window_gains(
    evaluation_rows, performance_rows=None, contaminated_power_tol: float = 0.5
):
    """Per-window gains plus a contamination mask aligned to ``evaluation_rows``."""
    contaminated = _contaminated_mask(evaluation_rows, contaminated_power_tol)
    if performance_rows:
        baseline = float(np.median([float(r["edp_w_s2"]) for r in performance_rows]))
        usable = list(range(len(evaluation_rows)))
    else:
        usable = [
            i for i, r in enumerate(evaluation_rows)
            if _finite(r.get("reference_edp_w_s2"))
            and float(r["reference_edp_w_s2"]) > 0.0
        ]
        if not usable:
            return [], contaminated
        baseline = float(
            np.median([float(evaluation_rows[i]["reference_edp_w_s2"]) for i in usable])
        )
    if not np.isfinite(baseline) or baseline <= 0.0:
        return [], contaminated
    gains = {}
    for i in usable:
        edp = evaluation_rows[i].get("edp_w_s2")
        if not _finite(edp):
            continue
        gains[i] = (baseline - float(edp)) / baseline
    return gains, contaminated


def _rows_at_lambda(rows, lam, tol=1e-9):
    """Rows whose ``requested_lambda`` matches ``lam``."""
    return [r for r in rows if abs(float(r["requested_lambda"]) - lam) < tol]


def _estimate_window_power(
    domain_power: float,
    other_cores_idle_power_w: float,
    basis: str,
    floor_w: float,
) -> Tuple[float, bool]:
    """Power term for one window's EDP, plus whether the floor was hit.

    The naive ``domain_power - other_cores_idle`` marginal estimate is the
    difference of two nearly equal quantities: the target core contributes only
    a fraction of a watt to a package that idles around 1.8 W, so the residual
    is dominated by the *other* cores' leakage drift rather than by the action
    under test.  On a 12-core package that estimator showed a within-lambda CV
    of ~0.77 and clamped to the floor for a large minority of windows, which
    swamped the few-percent differences between configurations.

    ``package`` (the default) keeps the raw RAPL power, whose within-lambda CV
    is ~0.15 and which is therefore the only basis able to rank configs at all.
    Common-mode drift is removed separately, when scoring, by comparing each
    window against its adjacent paired reference -- see ``_paired_ratio``.
    """
    if basis == "target_core":
        estimate = domain_power - other_cores_idle_power_w
    else:
        estimate = domain_power
    clamped = not (estimate > floor_w)
    return (max(float(floor_w), float(estimate)), bool(clamped))


def _planned_total_windows(args, eval_lambda_count: int) -> int:
    """Total windows the client must be provisioned for.

    Evaluation and per-lambda performance baselines are each run once per
    evaluated lambda, so they are multiplied by ``eval_lambda_count``; the
    initial reference and the initial performance phase are one-off and are not
    multiplied.  Getting this wrong under-provisions the client and lets it
    RESET mid-run, which invalidates the later windows.
    """
    per_lambda_eval = args.evaluation_windows * eval_lambda_count
    paired_reference = (
        args.evaluation_windows * eval_lambda_count
        if args.paired_evaluation_reference
        else 0
    )
    interleaved = (
        args.training_windows // args.interleave_reference_every
        if args.interleave_reference_every > 0
        else 0
    )
    return (
        args.reference_windows
        + args.training_windows
        + per_lambda_eval
        + args.performance_reference_windows
        + args.performance_reference_per_lambda_windows * eval_lambda_count
        + interleaved
        + paired_reference
    )


def _per_lambda_baseline_summary(
    evaluation_rows, performance_rows, eval_lambdas, contaminated_power_tol: float = 0.5
):
    """Per-lambda ondemand and performance baselines plus per-lambda gains.

    The pooled ``evaluation_reference_median_edp_w_s2`` /
    ``performance_reference_median_edp_w_s2`` values mix every lambda into one
    median, so they cannot be compared against a single lambda's EDP.  These are
    the values that must be used for per-lambda reporting.

    A lambda appears in the performance dicts only if it actually has
    performance-governor windows, so a missing baseline can never silently fall
    back to the ondemand one.

    Gains are reported as a median plus a trimmed mean and split by whether the
    window's power looked like a RAPL outlier for that lambda, because a single
    spike can move a plain mean by hundreds of percent.
    """
    ref_medians, deltas = {}, {}
    ref_gains, ref_gains_clean = {}, {}
    perf_medians, perf_gains, perf_gains_clean = {}, {}, {}
    contaminated_counts, ev_counts = {}, {}

    for lam in eval_lambdas:
        ev = _rows_at_lambda(evaluation_rows, lam)
        if not ev:
            continue
        paired = [
            float(r["reference_edp_w_s2"]) for r in ev
            if _finite(r.get("reference_edp_w_s2"))
            and float(r["reference_edp_w_s2"]) > 0.0
        ]
        if paired:
            ref_medians[float(lam)] = float(np.median(paired))
            deltas[float(lam)] = float(
                np.median([
                    float(r["edp_w_s2"]) / float(r["reference_edp_w_s2"]) - 1.0
                    for r in ev
                    if _finite(r.get("reference_edp_w_s2"))
                    and float(r["reference_edp_w_s2"]) > 0.0
                ])
            )
            gains, contaminated = _per_window_gains(ev, contaminated_power_tol=contaminated_power_tol)
            ref_gains[float(lam)] = _robust_gain([gains[i] for i in sorted(gains)])
            clean = [gains[i] for i in sorted(gains) if not contaminated[i]]
            ref_gains_clean[float(lam)] = _robust_gain(clean)
            contaminated_counts[float(lam)] = int(sum(contaminated))
            ev_counts[float(lam)] = int(len(ev))

        perf = _rows_at_lambda(performance_rows, lam)
        if perf:
            perf_medians[float(lam)] = float(
                np.median([float(r["edp_w_s2"]) for r in perf])
            )
            gains, contaminated = _per_window_gains(ev, perf, contaminated_power_tol)
            perf_gains[float(lam)] = _robust_gain([gains[i] for i in sorted(gains)])
            clean = [gains[i] for i in sorted(gains) if not contaminated[i]]
            perf_gains_clean[float(lam)] = _robust_gain(clean)

    return {
        "evaluation_per_lambda_reference_median_edp_w_s2": ref_medians or None,
        "evaluation_per_lambda_median_delta_vs_reference": deltas or None,
        "evaluation_per_lambda_gain_vs_reference_median": ref_gains or None,
        "evaluation_per_lambda_gain_vs_reference_median_clean": ref_gains_clean or None,
        "evaluation_per_lambda_performance_median_edp_w_s2": perf_medians or None,
        "evaluation_per_lambda_gain_vs_performance_median": perf_gains or None,
        "evaluation_per_lambda_gain_vs_performance_median_clean": (
            perf_gains_clean or None
        ),
        "evaluation_per_lambda_contaminated_windows": contaminated_counts or None,
        "evaluation_per_lambda_total_windows": ev_counts or None,
        "contaminated_power_tol": contaminated_power_tol,
    }


def _gain_vs_baseline(evaluation_rows, performance_rows=None):
    """Mean EDP gain of evaluation windows vs a stable baseline median.

    Positive gain means the model beat the baseline.  With
    ``performance_rows`` the baseline is the median EDP of the
    performance-governor windows; otherwise it is the median of the paired
    ondemand reference EDPs from the evaluation phase itself, and only windows
    that actually carry a finite positive paired reference are counted.
    """
    if performance_rows:
        baseline = float(
            np.median([float(r["edp_w_s2"]) for r in performance_rows])
        )
        usable = list(evaluation_rows)
    else:
        usable = [
            r for r in evaluation_rows
            if isinstance(r.get("reference_edp_w_s2"), (int, float))
            and np.isfinite(float(r["reference_edp_w_s2"]))
            and float(r["reference_edp_w_s2"]) > 0.0
        ]
        if not usable:
            return None
        baseline = float(
            np.median([float(r["reference_edp_w_s2"]) for r in usable])
        )
    if not np.isfinite(baseline) or baseline <= 0.0:
        return None
    gains = [(baseline - float(r["edp_w_s2"])) / baseline for r in usable]
    return {
        "mean_gain": float(np.mean(gains)),
        "win_rate": float(np.mean(np.asarray(gains) > 0.0)),
        "min_gain": float(np.min(gains)),
        "max_gain": float(np.max(gains)),
        "n_windows": float(len(gains)),
    }


def _gain_vs_baseline_gflops(evaluation_rows, baseline_rows):
    """Mean GFLOPS/W gain of evaluation windows vs a stable baseline median.

    Positive gain means the model beat the baseline. Higher GFLOPS/W is better.
    """
    baseline = float(np.median([float(r["gflops_per_w"]) for r in baseline_rows]))
    if not np.isfinite(baseline) or baseline <= 0.0:
        return None
    gains = [(float(r["gflops_per_w"]) - baseline) / baseline for r in evaluation_rows]
    return {
        "mean_gain": float(np.mean(gains)),
        "win_rate": float(np.mean(np.asarray(gains) > 0.0)),
        "min_gain": float(np.min(gains)),
        "max_gain": float(np.max(gains)),
        "n_windows": float(len(gains)),
    }


class PhaseTracker:
    """Phase-based action re-decision based on worker-core CPU utilization.

    A "phase" holds one chosen action.  The phase baseline is the mean
    utilization observed during the first --phase-baseline-s after the action is
    applied; the phase ends (re-deciding the action) when the utilization over
    the most recent --phase-recent-s drifts more than
    --phase-util-threshold-pct from the baseline -- but never before
    --phase-min-s, and never later than --phase-max-s even if utilization is
    flat.  Reference windows always use fixed-window timing regardless of the
    decision trigger so the EDP baseline stays comparable.
    """

    def __init__(
        self,
        baseline_s: float,
        recent_s: float,
        threshold_pct: float,
        min_s: float,
        max_s: float,
    ):
        self.baseline_s = float(baseline_s)
        self.recent_s = float(recent_s)
        self.threshold = float(threshold_pct) / 100.0
        self.min_s = float(min_s)
        self.max_s = float(max_s)
        self._samples: List[Tuple[float, float]] = []
        self._baseline: Optional[float] = None

    def reset(self) -> None:
        self._samples = []
        self._baseline = None

    def add_sample(self, elapsed_s: float, utilization: float) -> None:
        self._samples.append((float(elapsed_s), float(utilization)))

    def _satisfied_mean(self, start_s: float, end_s: float) -> Optional[float]:
        usable = [
            utilization
            for elapsed, utilization in self._samples
            if start_s <= elapsed < end_s
        ]
        if not usable:
            return None
        return float(np.mean(usable))

    def should_end(self, elapsed_s: float) -> bool:
        if elapsed_s >= self.max_s:
            return True
        if elapsed_s < self.min_s:
            return False
        recent = self._satisfied_mean(elapsed_s - self.recent_s, elapsed_s)
        if self._baseline is None:
            baseline = self._satisfied_mean(0.0, self.baseline_s)
            if baseline is None:
                return False
            self._baseline = baseline
        if recent is None:
            return False
        return abs(recent - self._baseline) >= self.threshold

    @property
    def baseline_utilization(self) -> Optional[float]:
        return self._baseline

    @property
    def recent_utilization(self) -> Optional[float]:
        if not self._samples:
            return None
        elapsed = self._samples[-1][0]
        usable = [
            utilization
            for sample_elapsed, utilization in self._samples
            if sample_elapsed >= max(0.0, elapsed - self.recent_s)
        ]
        if not usable:
            return None
        return float(np.mean(usable))


def _state_from_measurement(
    agent, row, sampling_rate, threshold, reference_power, progress
) -> np.ndarray:
    return agent.encode_observation(
        float(row["utilization"]),
        int(sampling_rate),
        int(threshold),
        float(row["mean_measured_frequency_khz"]),
        float(row["estimated_target_power_w"]) / max(reference_power, 1e-9),
        float(progress),
    )


def _measurement_as_candidate_row(
    measured: Dict[str, object], aggregate: Dict[str, float], edp: float
) -> Dict[str, object]:
    return {
        "utilization": measured["utilization"],
        "mean_measured_frequency_khz": measured["mean_measured_frequency_khz"],
        "estimated_target_power_w": measured["estimated_target_power_w"],
        "edp_w_s2": edp,
    }


def _base_row(
    *,
    agent,
    run_number,
    phase,
    config_id,
    sampling_rate,
    threshold,
    frequency_id,
    requested_frequency,
    mapped_frequency,
    measured: Dict[str, object],
    aggregate: Dict[str, float],
    edp: float,
    reward: float,
    done: bool,
    reference_candidate: bool,
    reference_edp,
    reference_source,
    reference_samples,
    epsilon,
    exploratory,
    greedy_config_id,
    greedy_frequency_bin_id,
    chosen_q,
    max_q,
    training_loss,
    training_updates,
    training_metrics: Optional[Dict[str, object]] = None,
    episode_id: str = "",
    episode_step: str = "",
    requested_lambda: object = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "phase": phase,
        "run": run_number,
        "episode_id": episode_id,
        "episode_step": episode_step,
        "requested_lambda": requested_lambda,
        "timestamp": time.time(),
        "config_id": config_id,
        "sampling_rate_us": sampling_rate,
        "threshold_pct": threshold,
        "frequency_bin_id": frequency_id,
        "requested_frequency_khz": requested_frequency,
        "mapped_frequency_khz": mapped_frequency,
        "jobs_completed": aggregate.get("jobs_completed", ""),
        "mean_response_time_s": aggregate.get("mean_response_time_s", ""),
        "mean_service_time_s": aggregate.get("mean_service_time_s", ""),
        "mean_wait_time_s": aggregate.get("mean_wait_time_s", ""),
        "mean_measured_frequency_khz": measured["mean_measured_frequency_khz"],
        "utilization": measured["utilization"],
        "epsilon": epsilon,
        "exploratory": exploratory,
        "greedy_config_id": greedy_config_id,
        "greedy_frequency_bin_id": greedy_frequency_bin_id,
        "chosen_q": chosen_q,
        "max_q": max_q,
        "window_seconds": measured["window_seconds"],
        "arrival_rate_jobs_per_s": (
            aggregate["jobs_completed"] / max(measured["window_seconds"], 1e-9)
            if aggregate.get("jobs_completed")
            else ""
        ),
        "rapl_domain": measured.get("rapl_domain", ""),
        "domain_energy_j": measured["domain_energy_j"],
        "domain_power_w": measured["domain_power_w"],
        "other_cores_idle_power_w": measured["other_cores_idle_power_w"],
        "estimated_target_power_w": measured["estimated_target_power_w"],
        "power_basis": measured["power_basis"],
        "power_clamped": measured.get("power_clamped", ""),
        "temperature_c": measured["temperature_c"],
        "decision_trigger": measured.get("decision_trigger", ""),
        "phase_baseline_utilization": measured.get("phase_baseline_utilization", ""),
        "phase_recent_utilization": measured.get("phase_recent_utilization", ""),
        "edp_w_s2": edp if np.isfinite(edp) else "",
        "reference_edp_w_s2": "" if reference_edp is None else float(reference_edp),
        "reference_baseline_source": reference_source,
        "reference_baseline_samples": reference_samples,
        "mean_flops_per_job": aggregate.get("mean_flops_per_job", ""),
        "total_flops": aggregate.get("total_flops", ""),
        "gflops_per_s": aggregate.get("gflops_per_s", ""),
        "gflops_per_w": aggregate.get("gflops_per_w", ""),
        "reward": reward,
        "done": done,
        "replay_size": agent.replay.size,
        "training_loss": training_loss,
        "training_updates": training_updates,
        "training_td_error": (training_metrics or {}).get("td_error", ""),
        "training_td_error_abs": (training_metrics or {}).get("td_error_abs", ""),
        "training_target_q_mean": (training_metrics or {}).get("target_q_mean", ""),
        "training_chosen_q_mean": (training_metrics or {}).get("chosen_q_mean", ""),
        "training_max_q_mean": (training_metrics or {}).get("max_q_mean", ""),
        "training_q_gap_mean": (training_metrics or {}).get("q_gap_mean", ""),
        "training_greedy_rate": (training_metrics or {}).get("greedy_rate", ""),
        "telemetry_path": "",
        "server_pid": "",
        "client_pid": "",
    }
    return row


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print(f"Interrupted: {error}", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)