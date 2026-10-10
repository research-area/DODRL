#!/usr/bin/env python3
"""Shared helpers for the MG1 joint config+frequency runner.

Workload-neutral utilities extracted from the joint training runner:
CSV row logging, package idle-power calibration, the log-space EDP reward,
per-window training-episode position, and state attachment for run rows.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cpu_control import CpuFrequencyController, RaplDomain


class CsvAppender:
    def __init__(self, path: Path, fields: List[str]):
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, object]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self.fields})


def calibrate_package_idle_power(
    cpu: CpuFrequencyController,
    rapl: RaplDomain,
    seconds: float,
    n_package_cores: int,
) -> Tuple[float, float, float]:
    """Measure package idle power and derive per-core idle attribution."""
    cpu.set_target_userspace_fixed(cpu.requested_min_khz)
    before = rapl.read()
    after_timestamp = before.timestamp + seconds
    while time.monotonic() < after_timestamp:
        time.sleep(0.05)
    after = rapl.read()
    elapsed = after.timestamp - before.timestamp
    if elapsed <= 0:
        raise RuntimeError("Invalid package-idle-calibration interval")
    package_idle_power_w = rapl.joules_between(before, after) / elapsed
    target_idle_power_w = package_idle_power_w / max(1, n_package_cores)
    other_cores_idle_power_w = package_idle_power_w - target_idle_power_w
    cpu.restore_target_ondemand()
    return package_idle_power_w, target_idle_power_w, other_cores_idle_power_w


def edp_reward(edp: float, reference_edp: float) -> float:
    if not np.isfinite(edp) or edp <= 0.0:
        raise ValueError("EDP must be finite and positive")
    if not np.isfinite(reference_edp) or reference_edp <= 0.0:
        raise ValueError("Reference EDP must be finite and positive")
    return float(np.clip(-math.log(edp / reference_edp), -5.0, 5.0))


def training_episode_position(
    run_number: int,
    total_runs: int,
    interleave_reference_every: int,
) -> Tuple[int, int, bool]:
    if run_number < 1 or total_runs < 1 or run_number > total_runs:
        raise ValueError("run_number must lie between 1 and total_runs")
    if interleave_reference_every < 0:
        raise ValueError("interleave_reference_every must be non-negative")
    if interleave_reference_every == 0:
        return 1, run_number, run_number == total_runs
    episode_id = (run_number - 1) // interleave_reference_every + 1
    episode_step = (run_number - 1) % interleave_reference_every + 1
    terminal = (
        run_number == total_runs
        or run_number % interleave_reference_every == 0
    )
    return episode_id, episode_step, terminal


def attach_state(row: Dict[str, object], prefix: str, state: Optional[np.ndarray]) -> None:
    for index in range(7):
        row[f"{prefix}_{index}"] = "" if state is None else float(state[index])
