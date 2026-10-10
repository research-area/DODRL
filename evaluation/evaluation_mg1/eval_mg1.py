#!/usr/bin/env python3
"""MG1 evaluation: fixed 5-min runs, duration-neutral metrics.

Reuses the EXACT MG1 launch + measurement machinery from run_joint_freq_mg1.py
(the same code path as training). Only difference: fixed-duration windows
(not phase-triggered) and duration-neutral metrics (power, energy-per-job).

Per lambda:
  - ondemand (default) 3x (averaged)
  - agent inference (closed-loop, re-decides every --decision-interval-s)

Metrics (all duration-neutral, no window-length confound):
  - power_w          = domain_energy_j / window_seconds
  - energy_per_job_j = domain_energy_j / jobs_completed
  - svc_s            = mean_service_time_s
  - mrt_s            = mean_response_time_s
  - jobs             = jobs_completed
  - mean_freq_khz    = mean_measured_frequency_khz

Results written to evaluation_mg1/results/<lambda_tag>/ and a summary.json
at evaluation_mg1/results/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- imports from the project ---
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "agent"))
from cpu_control import (
    CpuFrequencyController,
    RaplDomain,
    cpu_stat,
    online_cpu_count,
    read_max_temperature_c,
    utilization_between,
)
from config_only_ddqn import ConfigOnlyDDQN, ConfigOnlyWrapper
from sequential_ddqn import DDQNConfig
from config_only_ddqn import ACTION_GRID as BASE_ACTION_GRID
from sequential_ddqn import SAMPLING_RATES_US, THRESHOLDS_PCT
# Reuse the EXACT launch + measurement machinery from the training runner
from run_joint_freq_mg1 import (
    launch_session,
    TelemetryReader,
    make_environment,
    affinity_reserved_mask,
    Mg1Proc,
)

# Config-only action grid: (sr, thr) only — 50 actions
ACTION_GRID_CO = np.asarray(
    [(int(sr), int(thr)) for sr in SAMPLING_RATES_US for thr in THRESHOLDS_PCT],
    dtype=np.int64,
)

MG1_FLOPS_PER_ITERATION = 6
LAMBDAS = [0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 7.5]


def summarise(rows):
    """Summarise telemetry rows. Handles both dict rows (from our TelemetryReader)
    and list rows (from the runner's TelemetryReader which returns lists)."""
    if not rows:
        return None
    n = len(rows)
    # Detect format: dict or list
    if isinstance(rows[0], dict):
        total_resp = sum(r["response_ns"] for r in rows)
        total_calc = sum(r["calc_ns"] for r in rows)
        total_wait = sum(r["wait_ns"] for r in rows)
        total_flops = sum(r.get("repeat", 0) for r in rows) * MG1_FLOPS_PER_ITERATION
    else:
        # List format: [wall_ms, response_ns, calc_ns, wait_ns, cpu_ns, repeat]
        total_resp = sum(r[1] for r in rows)
        total_calc = sum(r[2] for r in rows)
        total_wait = sum(r[3] for r in rows)
        total_flops = sum(r[5] for r in rows) * MG1_FLOPS_PER_ITERATION
    return {
        "jobs_completed": n,
        "mean_response_time_s": total_resp / n / 1e9,
        "mean_service_time_s": total_calc / n / 1e9,
        "mean_wait_time_s": total_wait / n / 1e9,
        "total_flops": total_flops,
        "mean_flops_per_job": total_flops / n if n else 0.0,
    }


def load_agent(model_path: Path) -> ConfigOnlyWrapper:
    config = DDQNConfig(
        state_dim=7, n_frequency_bins=1, gamma=0.95,
        learning_rate=1e-3, batch_size=16, min_replay_size=16,
        seed=42, fmin_khz=800_000, fmax_khz=2_100_000,
    )
    ddqn = ConfigOnlyDDQN(config)
    ddqn.load(model_path)
    return ConfigOnlyWrapper(ddqn)


def encode_state(agent, util, sr, thr, freq_khz, ref_power):
    return agent.encode_observation(
        util, int(sr), int(thr), float(freq_khz),
        1.0,  # power_ratio — neutral for eval
        1.0,  # progress
    )


def run_fixed_window(
    *, cpu: CpuFrequencyController, rapl: RaplDomain,
    proc, telemetry: TelemetryReader,
    duration_s: float, target_cpu: int,
    label: str,
) -> Dict:
    """Run a fixed-duration measurement window using the same RAPL/telemetry
    machinery as run_joint_freq_mg1.py:956-1080."""
    util_before = cpu_stat(target_cpu)
    energy_before = rapl.read()
    start = time.monotonic()
    freq_samples: List[float] = []
    window_rows: List[Dict] = []

    while True:
        time.sleep(0.2)
        if not proc.alive():
            raise RuntimeError(f"MG1 server/client exited during {label}")
        freq_samples.append(float(cpu.current_frequency_khz()))
        rows, _ = telemetry.read()
        window_rows.extend(rows)
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break

    energy_after = rapl.read()
    util_after = cpu_stat(target_cpu)
    window_time = energy_after.timestamp - energy_before.timestamp
    domain_energy = rapl.joules_between(energy_before, energy_after)
    domain_power = domain_energy / max(window_time, 1e-9)
    util = utilization_between(util_before, util_after)
    agg = summarise(window_rows)

    if agg is None or agg["jobs_completed"] == 0:
        return {"valid": False, "label": label, "window_seconds": window_time}

    jobs = agg["jobs_completed"]
    return {
        "valid": True,
        "label": label,
        "window_seconds": window_time,
        "jobs_completed": jobs,
        "mean_response_time_s": agg["mean_response_time_s"],
        "mean_service_time_s": agg["mean_service_time_s"],
        "mean_wait_time_s": agg["mean_wait_time_s"],
        "mean_measured_frequency_khz": float(np.mean(freq_samples)) if freq_samples else 0.0,
        "utilization": util,
        "domain_energy_j": domain_energy,
        "domain_power_w": domain_power,
        "power_w": domain_power,
        "energy_per_job_j": domain_energy / jobs,
        "total_flops": agg["total_flops"],
        "gflops_per_s": agg["total_flops"] / max(window_time, 1e-9) / 1e9,
    }


def run_agent_window(
    *, cpu: CpuFrequencyController, rapl: RaplDomain,
    proc, telemetry: TelemetryReader,
    duration_s: float, target_cpu: int,
    agent: ConfigOnlyWrapper, reference_power_w: float,
    decision_interval_s: float,
    label: str,
) -> Dict:
    """Run a fixed-duration window with the agent re-deciding every decision_interval_s."""
    util_before = cpu_stat(target_cpu)
    energy_before = rapl.read()
    start = time.monotonic()
    freq_samples: List[float] = []
    window_rows: List[Dict] = []
    decisions = 0

    current_sr, current_thr = 10_000, 95
    cpu.set_config_only(current_sr, current_thr)

    while True:
        time.sleep(0.2)
        if not proc.alive():
            raise RuntimeError(f"MG1 server/client exited during {label}")
        freq_samples.append(float(cpu.current_frequency_khz()))
        rows, _ = telemetry.read()
        window_rows.extend(rows)
        elapsed = time.monotonic() - start

        if elapsed >= (decisions + 1) * decision_interval_s:
            util_now = cpu_stat(target_cpu)
            util = utilization_between(util_before, util_now)
            state = encode_state(agent, util, current_sr, current_thr,
                                 float(cpu.current_frequency_khz()), reference_power_w)
            config_id, _, _, greedy_id, _, _, _ = agent.action(state, epsilon=0.0)
            sr, thr = ACTION_GRID_CO[config_id]
            cpu.set_config_only(int(sr), int(thr))
            current_sr, current_thr = int(sr), int(thr)
            decisions += 1

        if elapsed >= duration_s:
            break

    energy_after = rapl.read()
    util_after = cpu_stat(target_cpu)
    window_time = energy_after.timestamp - energy_before.timestamp
    domain_energy = rapl.joules_between(energy_before, energy_after)
    domain_power = domain_energy / max(window_time, 1e-9)
    util = utilization_between(util_before, util_after)
    agg = summarise(window_rows)

    if agg is None or agg["jobs_completed"] == 0:
        return {"valid": False, "label": label, "window_seconds": window_time}

    jobs = agg["jobs_completed"]
    return {
        "valid": True,
        "label": label,
        "window_seconds": window_time,
        "jobs_completed": jobs,
        "mean_response_time_s": agg["mean_response_time_s"],
        "mean_service_time_s": agg["mean_service_time_s"],
        "mean_wait_time_s": agg["mean_wait_time_s"],
        "mean_measured_frequency_khz": float(np.mean(freq_samples)) if freq_samples else 0.0,
        "utilization": util,
        "domain_energy_j": domain_energy,
        "domain_power_w": domain_power,
        "power_w": domain_power,
        "energy_per_job_j": domain_energy / jobs,
        "total_flops": agg["total_flops"],
        "gflops_per_s": agg["total_flops"] / max(window_time, 1e-9) / 1e9,
        "decisions": decisions,
        "final_sr": current_sr,
        "final_thr": current_thr,
    }


def main():
    parser = argparse.ArgumentParser(description="MG1 fixed-window evaluation")
    parser.add_argument("--target-cpu", type=int, default=3)
    parser.add_argument("--controller-cpu", type=int, default=10)
    parser.add_argument("--client-cpu", type=int, default=7)
    parser.add_argument("--mg1-root", type=Path, default=Path("MG1"))
    parser.add_argument("--models-root", type=Path,
                        default=_REPO / "models")
    parser.add_argument("--results", type=Path, default=Path("evaluation_mg1/results"))
    parser.add_argument("--duration-s", type=float, default=300.0,
                        help="Duration of each run in seconds (default 300 = 5 min)")
    parser.add_argument("--decision-interval-s", type=float, default=5.0,
                        help="Agent re-decides every N seconds (default 5)")
    parser.add_argument("--baseline-reps", type=int, default=3,
                        help="Reps per ondemand baseline")
    parser.add_argument("--agent-reps", type=int, default=10,
                        help="Agent reps per lambda")
    parser.add_argument("--lambdas", type=str, default="",
                        help="Comma-separated lambdas (default: all 10)")
    parser.add_argument("--n-package-cores", type=int, default=12)
    parser.add_argument("--rapl-domain", type=str, default="core")
    parser.add_argument("--min-frequency-khz", type=int, default=800_000)
    parser.add_argument("--max-frequency-khz", type=int, default=2_100_000)
    parser.add_argument("--warmup-s", type=float, default=15.0)
    parser.add_argument("--client-repeat", type=int, default=1_000_000)
    parser.add_argument("--temperature-limit-c", type=float, default=85.0)
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else LAMBDAS
    args.mg1_root = args.mg1_root.resolve()
    args.models_root = args.models_root.resolve()
    results_dir = args.results.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check loopback alias
    lo = subprocess.run(["ip", "addr", "show", "dev", "lo"],
                        capture_output=True, text=True).stdout
    if "10.0.0.2" not in lo:
        print("[setup] Adding 10.0.0.2/24 to loopback...")
        subprocess.run(["sudo", "ip", "addr", "add", "10.0.0.2/24", "dev", "lo"],
                       capture_output=True)

    # Check MG1 build
    build = args.mg1_root / "build"
    if not (build / "Server" / "Server.class").exists():
        print(f"ERROR: MG1 Server.class not found at {build}/Server/Server.class")
        sys.exit(1)
    if not (build / "client" / "LoadGenerator.class").exists():
        print(f"ERROR: MG1 LoadGenerator.class not found at {build}/client/LoadGenerator.class")
        sys.exit(1)
    if not (args.mg1_root / "Server" / "Server" / "lib").exists():
        print(f"ERROR: MG1 lib dir not found at {args.mg1_root}/Server/Server/lib")
        sys.exit(1)

    # Kill stale java
    subprocess.run(["pkill", "-9", "-f", "Server.Server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "LoadGenerator"], capture_output=True)
    time.sleep(1)

    cpu = CpuFrequencyController(
        args.target_cpu, args.controller_cpu,
        requested_min_khz=args.min_frequency_khz,
        requested_max_khz=args.max_frequency_khz,
    )
    cpu.assert_process_pinned()
    rapl = RaplDomain.discover(preference=args.rapl_domain)

    # Idle calibration
    print("[setup] Calibrating idle power...")
    time.sleep(2)
    idle_before = rapl.read()
    time.sleep(5)
    idle_after = rapl.read()
    idle_energy = rapl.joules_between(idle_before, idle_after)
    idle_power = idle_energy / 5.0
    print(f"  Idle power: {idle_power:.4f} W")

    all_results = {}

    for lam in lambdas:
        lam_tag = f"lam{str(lam).replace('.', '_')}"
        lam_dir = results_dir / lam_tag
        lam_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = lam_dir / "telemetry.csv"

        print(f"\n{'='*60}")
        print(f" λ={lam} | duration={args.duration_s}s | reps={args.baseline_reps}")
        print(f"{'='*60}")

        # Load agent model — DODRL uses lambda_X_Y naming
        # DODRL model dirs: lambda_0_8, lambda_1_6, ... (underscore-separated)
        lam_dir_name = f"lambda_{str(lam).replace('.', '_')}"
        model_path = args.models_root / lam_dir_name / "mg1_ddqn_final.npz"
        if not model_path.exists():
            print(f"  WARNING: model not found at {model_path}, skipping agent")
            agent = None
        else:
            agent = load_agent(model_path)
            print(f"  Loaded agent model: {model_path}")

        lam_results = {"ondemand": [], "agent": []}

        for rep in range(1, args.baseline_reps + 1):
            # --- ondemand ---
            print(f"  [ondemand {rep}/{args.baseline_reps}] launching MG1...")
            proc, _ = launch_session(
                root=args.mg1_root, server_cpu=args.target_cpu,
                client_cpu=args.client_cpu, arrival_rate=lam,
                duration_s=args.duration_s + 60,
                client_duration_ms=int((args.duration_s + 120) * 1000),
                client_repeat=args.client_repeat,
                telemetry_path=telemetry_path, log_dir=lam_dir,
            )
            time.sleep(args.warmup_s)
            telemetry = TelemetryReader(telemetry_path)
            telemetry.reset()
            telemetry.flush()

            cpu.prepare_config_only(10_000, 95)
            r = run_fixed_window(
                cpu=cpu, rapl=rapl, proc=proc, telemetry=telemetry,
                duration_s=args.duration_s, target_cpu=args.target_cpu,
                label=f"ondemand_{rep}",
            )
            r["rep"] = rep
            r["mode"] = "ondemand"
            lam_results["ondemand"].append(r)
            print(f"    jobs={r.get('jobs_completed', 0)} P={r.get('power_w', 0):.3f}W "
                  f"E/job={r.get('energy_per_job_j', 0):.4f}J svc={r.get('mean_service_time_s', 0)*1000:.1f}ms")
            proc.terminate()
            time.sleep(2)

        # --- agent ---
        if agent is not None:
            for rep in range(1, args.agent_reps + 1):
                print(f"  [agent {rep}/{args.agent_reps}] launching MG1...")
                proc, _ = launch_session(
                    root=args.mg1_root, server_cpu=args.target_cpu,
                    client_cpu=args.client_cpu, arrival_rate=lam,
                    duration_s=args.duration_s + 60,
                    client_duration_ms=int((args.duration_s + 120) * 1000),
                    client_repeat=args.client_repeat,
                    telemetry_path=telemetry_path, log_dir=lam_dir,
                )
                time.sleep(args.warmup_s)
                telemetry = TelemetryReader(telemetry_path)
                telemetry.reset()
                telemetry.flush()

                cpu.prepare_config_only(10_000, 95)
                r = run_agent_window(
                    cpu=cpu, rapl=rapl, proc=proc, telemetry=telemetry,
                    duration_s=args.duration_s, target_cpu=args.target_cpu,
                    agent=agent, reference_power_w=idle_power,
                    decision_interval_s=args.decision_interval_s,
                    label=f"agent_{rep}",
                )
                r["rep"] = rep
                r["mode"] = "agent"
                lam_results["agent"].append(r)
                print(f"    jobs={r.get('jobs_completed', 0)} P={r.get('power_w', 0):.3f}W "
                      f"E/job={r.get('energy_per_job_j', 0):.4f}J svc={r.get('mean_service_time_s', 0)*1000:.1f}ms "
                      f"dec={r.get('decisions', 0)} sr={r.get('final_sr', 0)} thr={r.get('final_thr', 0)}")
                proc.terminate()
                time.sleep(2)

        # Write per-lambda CSV
        csv_path = lam_dir / "runs.csv"
        all_rows = []
        for mode in ("ondemand", "agent"):
            all_rows.extend(lam_results[mode])
        if all_rows:
            # Union of all keys (agent rows have extra fields)
            all_keys = []
            for row in all_rows:
                for k in row:
                    if k not in all_keys:
                        all_keys.append(k)
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                w.writerows(all_rows)

        all_results[lam] = lam_results
        print(f"  λ={lam} done. Results: {csv_path}")

    # Write summary
    summary = {}
    for lam, res in all_results.items():
        lam_tag = f"lam{str(lam).replace('.', '_')}"
        entry = {}
        for mode in ("ondemand", "agent"):
            valid = [r for r in res[mode] if r.get("valid")]
            if not valid:
                entry[mode] = {"valid": False}
                continue
            entry[mode] = {
                "valid": True,
                "n": len(valid),
                "power_w": float(np.mean([r["power_w"] for r in valid])),
                "energy_per_job_j": float(np.mean([r["energy_per_job_j"] for r in valid])),
                "svc_s": float(np.mean([r["mean_service_time_s"] for r in valid])),
                "mrt_s": float(np.mean([r["mean_response_time_s"] for r in valid])),
                "jobs": float(np.mean([r["jobs_completed"] for r in valid])),
                "mean_freq_khz": float(np.mean([r["mean_measured_frequency_khz"] for r in valid])),
                "utilization": float(np.mean([r["utilization"] for r in valid])),
            }
        # Gains vs default ondemand: + = agent better (lower is better for all these metrics)
        if entry.get("agent", {}).get("valid") and entry.get("ondemand", {}).get("valid"):
            for metric in ("power_w", "energy_per_job_j", "svc_s"):
                od = entry["ondemand"][metric]
                ag = entry["agent"][metric]
                entry[f"gain_{metric}_vs_od"] = 100.0 * (1.0 - ag / od) if od > 0 else 0.0
        summary[lam_tag] = entry

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f" Summary written to {summary_path}")
    print(f"{'='*60}")

    # Print table
    print(f"\n{'λ':<5} {'mode':<12} {'P(W)':>7} {'E/job(J)':>9} {'svc(ms)':>8} {'jobs':>6} {'freq(kHz)':>10}")
    for lam in lambdas:
        lam_tag = f"lam{str(lam).replace('.', '_')}"
        for mode in ("ondemand", "agent"):
            e = summary.get(lam_tag, {}).get(mode, {})
            if e.get("valid"):
                print(f"{lam:<5} {mode:<12} {e['power_w']:>7.3f} {e['energy_per_job_j']:>9.4f} "
                      f"{e['svc_s']*1000:>8.1f} {e['jobs']:>6.0f} {e['mean_freq_khz']:>10.0f}")
            else:
                print(f"{lam:<5} {mode:<12} {'N/A':>7} {'N/A':>9} {'N/A':>8} {'N/A':>6} {'N/A':>10}")

    # Kill any remaining java
    subprocess.run(["pkill", "-9", "-f", "Server.Server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "LoadGenerator"], capture_output=True)


if __name__ == "__main__":
    main()
